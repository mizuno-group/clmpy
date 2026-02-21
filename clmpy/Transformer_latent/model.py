import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Tuple, Optional

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class Conv1D(nn.Module):
    def __init__(self, out_dim, in_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.weight.size(1), )
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(*size_out)

# --- RPE Implementation Start ---

class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, n_head=12):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_head = n_head
        self.relative_attention_bias = nn.Embedding(num_buckets, n_head)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Adapted from T5/Mesh Tensorflow.
        Translate relative position to a bucket number for efficient lookup.
        """
        ret = 0
        n = -relative_position
        if num_buckets > 0:
            ret += (n < 0).to(torch.long) * num_buckets // 2 # Offset for negative values
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_length, key_length, device):
        """
        Compute binned relative position bias.
        """
        # Generate position indices
        # query_pos: [query_length]
        # key_pos:   [key_length]
        # We need to handle the case where we have cached past keys.
        # The query usually starts *after* the past keys.
        
        # In a standard forward pass without past: q=0..L, k=0..L
        # With past: q=L_past..L_past+L_q, k=0..L_past+L_q
        
        # Note: To simplify, we assume the 'offset' is handled by the caller or
        # we calculate relative distance based on the shapes. 
        # Here we construct a matrix of shape [q_len, k_len]
        
        context_position = torch.arange(key_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(query_length, dtype=torch.long, device=device)[None, :] 
        
        # If we are decoding, the query is at the END of the key sequence.
        # We need to know the offset. However, a simpler relative logic used in T5
        # is just (key_idx - query_idx).
        
        # To handle 'past' correctly (where query is at the end), we need to offset the query indices.
        # Offset = key_length - query_length
        offset = key_length - query_length
        memory_position = memory_position + offset
        
        relative_position = memory_position - context_position
        # Shape: [key_length, query_length] -> transpose to [query_length, key_length]
        relative_position = relative_position.transpose(0, 1)

        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        # Shape: [q_len, k_len, n_head] -> permute to [1, n_head, q_len, k_len]
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

# --- RPE Implementation End ---

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0)], requires_grad=False)
        return self.dropout(x)

class Attention(GPT2Attention):
    def __init__(self, config, scale=False):
        super().__init__(config)
        nx = config.embedding_dim
        self.n_head = config.n_head
        self.split_size = nx
        self.scale = scale
        self.head_dim = nx // self.n_head
        self.c_attn = Conv1D(3 * nx, nx)
        self.c_proj = Conv1D(nx, nx)
        self.attn_dropout = nn.Dropout(config.dropout)
        
        # --- RPE Integration ---
        self.use_rpe = getattr(config, "use_rpe", False)
        if self.use_rpe:
            # RPE Defaults if not provided in config
            rpe_buckets = getattr(config, "relative_attention_num_buckets", 32)
            rpe_max_dist = getattr(config, "relative_attention_max_distance", 128)
            
            self.rpe_bias = RelativePositionBias(
                num_buckets=rpe_buckets,
                max_distance=rpe_max_dist,
                n_head=self.n_head
            )

    def _split_heads(self, x, num_heads, head_dim):
        # x: [B, L, D]
        B, L, D = x.size()
        x = x.view(B, L, num_heads, head_dim)   # [B, L, H, D/H]
        return x.permute(0, 2, 1, 3)            # [B, H, L, D/H]

    def _merge_heads(self, x, num_heads, head_dim):
        # x: [B, H, L, D/H]
        B, H, L, d = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, H, D/H]
        return x.view(B, L, H * d)    
        
    def _attn(self, q, k, v, attention_mask=None, rpe_bias=None):
        # q, k, v: [B, H, L, D_head]
        # Transpose k for matmul: [B, H, D_head, L]
        w = torch.matmul(q, k.transpose(-2, -1)) 
        w = w / math.sqrt(v.size(-1))
        
        # --- Apply RPE Bias ---
        if rpe_bias is not None:
            # rpe_bias shape: [1, H, L_q, L_k]
            w = w + rpe_bias

        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        outputs = torch.matmul(w, v) # [B, H, L, D_head]
        return outputs, w
    
    def forward(self, x, attention_mask=None, layer_past=None):
        # x: [L, B, D] -> transpose to [B, L, D] for processing
        x = self.c_attn(x).transpose(0, 1) # [B, L, 3D]
        query, key, value = x.split(self.split_size, dim=2) # [B, L, D] each

        # Split heads: [B, H, L, D_head]
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)

        # Handle Past (Caching)
        if layer_past is not None:
            # Assuming layer_past stores [key, value] in shape [B, H, L, D_head]
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # Save present for next step (Keep shape [B, H, L, D_head])
        present = (key, value) 

        # --- Calculate RPE Bias ---
        rpe_bias_tensor = None
        if self.use_rpe:
            query_len = query.size(2)
            key_len = key.size(2)
            # The bias module handles the relative indexing
            rpe_bias_tensor = self.rpe_bias(query_len, key_len, device=query.device)

        # Attention Calculation
        # Pass rpe_bias_tensor to _attn
        a, attn_weights = self._attn(query, key, value, attention_mask, rpe_bias=rpe_bias_tensor) 

        # Merge heads
        a = self._merge_heads(a, self.n_head, self.head_dim) # [B, L, D]
        a = self.c_proj(a)
        a = self.attn_dropout(a)

        # Output formatting matching original: [L, B, D], present, weights
        outputs = [a.transpose(0, 1), present, attn_weights]
        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, config, scale=False):
        gpt2config = GPT2Config(**config.__dict__)
        gpt2config.n_embd = config.embedding_dim
        
        # Propagate RPE config
        if hasattr(config, 'use_rpe'):
            gpt2config.use_rpe = config.use_rpe
            gpt2config.relative_attention_num_buckets = getattr(config, 'relative_attention_num_buckets', 32)
            gpt2config.relative_attention_max_distance = getattr(config, 'relative_attention_max_distance', 128)

        super().__init__()
        nx = config.embedding_dim
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(gpt2config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4*nx, gpt2config)
    
    def forward(self, x, attention_mask=None, layer_past=None):
        output_attn = self.attn(self.ln_1(x), attention_mask, layer_past)
        a = output_attn[0]
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = [x] + output_attn[1:]
        return outputs

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        nx = config.embedding_dim
        self.use_rpe = getattr(config, "use_rpe", False)

        self.wte = nn.Embedding(config.vocab_size, nx)
        
        # If using RPE, we usually disable Absolute Positional Encoding
        if self.use_rpe:
            self.wpe = nn.Identity() 
        else:
            self.wpe = PositionalEncoding(nx, config.dropout, max_len=config.n_positions)
            
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        
        self.ln_mem1 = nn.LayerNorm(nx)
        self.ln_mem2 = nn.LayerNorm(nx)
        self.ln_mem3 = nn.LayerNorm(nx)
        self.fc_latent = nn.Linear(3*nx, config.embedding_dim)

    def create_enc_attention_mask(self, input_ids):
        pad_array = input_ids == 0
        pad_array_ = pad_array.transpose(0, 1).unsqueeze(1).unsqueeze(2)
        return pad_array, torch.where(pad_array_ == True, float("-inf"), 0.0)
    
    def memory_pool(self, memory, pad_array):
        pad_array = pad_array.unsqueeze(-1)
        masked = memory.masked_fill(pad_array, -torch.inf)
        padding_mask = ~pad_array
        mx = torch.max(masked, dim=0)[0]
        ave = torch.sum(memory * padding_mask, dim=0) / torch.sum(padding_mask, dim=0)
        first = memory[0]
        return torch.cat([self.ln_mem1(mx), self.ln_mem2(ave), self.ln_mem3(first)], dim=1)
    
    def forward(self, x, past=None):
        # x: [L, B]
        input_shape = x.size()
        x = x.view(-1, input_shape[-1])
        if past is None:
            past = [None] * len(self.h)
            
        input_embeds = self.wte(x)
        
        if self.use_rpe:
             hidden_states = self.drop(input_embeds)
        else:
             hidden_states = self.wpe(input_embeds)

        pad_array, attention_mask = self.create_enc_attention_mask(x)
        output_shape = input_shape + (hidden_states.size(-1),)

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
            
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        latent = self.memory_pool(hidden_states, pad_array)
        latent = self.fc_latent(latent)
        return torch.tanh(latent)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        nx = config.embedding_dim
        self.use_rpe = getattr(config, "use_rpe", False)

        self.wte = nn.Embedding(config.vocab_size, nx)
        
        if self.use_rpe:
            self.wpe = nn.Identity()
        else:
            self.wpe = PositionalEncoding(nx, config.dropout, max_len=config.n_positions)
            
        self.input_proj = nn.Linear(nx, nx, bias=False)
        self.h = nn.ModuleList([TransformerBlock(config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.output_fc = nn.Linear(nx, config.vocab_size)
        self.device = config.device

    def create_dec_attention_mask(self, input_ids):
        l, b = input_ids.size()
        pad_array = (input_ids == 0).transpose(0, 1).unsqueeze(1).unsqueeze(2)
        seq_array = torch.triu(torch.full((l, l), True, device=self.device), diagonal=1)
        seq_array = seq_array.unsqueeze(0).unsqueeze(1)
        res = torch.logical_or(pad_array, seq_array)
        return torch.where(res == True, float("-inf"), 0.0)
    
    def forward(self, x, latent, layer_past=None):
        if layer_past is None:
            past = [None] * len(self.h)
        else:
            past = layer_past 
            
        attention_mask = self.create_dec_attention_mask(x)
        input_embeds = self.wte(x)
        
        if self.use_rpe:
            hidden_states = input_embeds
        else:
            hidden_states = self.wpe(input_embeds)
            
        # Latent injection
        hidden_states = hidden_states + latent.unsqueeze(1).transpose(0, 1)

        presents = ()
        for i, (block, layer_past_item) in enumerate(zip(self.h, past)):
            outputs = block(hidden_states, layer_past=layer_past_item, attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
            presents = presents + (present,)
            
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.output_fc(hidden_states)
        return hidden_states 

# The wrappers (TransformerLatent, downstream_MLP, etc.) remain unchanged.
# Ensure your 'config' object passed to the model has 'use_rpe = True'.
class TransformerLatent(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self,src,tgt,past=None):
        latent = self.encoder(src)
        outputs = self.decoder(tgt,latent,layer_past=past)
        return outputs, latent
    
class downstream_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.activation = nn.ReLU()
        self.dropout_rate = config.dropout
        self.use_batch_norm = config.batch_norm
        self.use_layer_norm = config.layer_norm
        layer_dim = config.layer_dim
        layer_dim.insert(0, self.latent_dim)
        self.linear = nn.ModuleList([
            nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim)-1)
        ])
        if self.use_batch_norm:
            self.batch_norm = nn.ModuleList([
                nn.BatchNorm1d(layer_dim[i+1]) for i in range(len(layer_dim)-1)
            ])
        else:
            self.batch_norm = None

        if self.use_layer_norm:
            self.layer_norm = nn.ModuleList([
                nn.LayerNorm(layer_dim[i+1]) for i in range(len(layer_dim)-1)
            ])
        else:
            self.layer_norm = None

        if self.dropout_rate > 0:
            self.dropout = nn.ModuleList([
                nn.Dropout(self.dropout_rate) for _ in range(len(layer_dim)-1)
            ])
        else:
            self.dropout = None 
        self.classifier = nn.Linear(layer_dim[-1], 1)

    def forward(self, x):
        for i, v in enumerate(self.linear):
            x = v(x)
            x = self.activation(x) 
            if self.dropout: 
                x = self.dropout[i](x)
        x = self.classifier(x)
        return x
        
class TransformerLatent_MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.mlp = downstream_MLP(config)

    def forward(self,src,tgt,past=None):
        latent = self.encoder(src)
        out = self.decoder(tgt,latent,layer_past=past)
        out_d = self.mlp(latent)
        return out, out_d, latent