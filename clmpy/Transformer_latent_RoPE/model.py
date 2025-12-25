import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2Block
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary positional embedding to query and key tensors."""
    # q, k: [B, H, L, D]
    # cos, sin: [1, 1, L, D] -> broadcast to [B, H, L, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, n_positions: int = 2048, base: int = 10000, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.n_positions = n_positions
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=n_positions, device=self.inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# --- RoPE Implementation End ---

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
        
        # --- RoPE Integration ---
        # configにuse_ropeがない場合はFalseとする安全策
        self.use_rope = getattr(config, "use_rope", False)
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim ({self.head_dim}) must be even to use RoPE.")
            
            # rope_baseの取得（デフォルト10000）
            rope_base = getattr(config, "rope_base", 10000)
            self.rotary_emb = RotaryEmbedding(self.head_dim, n_positions=config.n_positions, base=rope_base)
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
        
    def _attn(self, q, k, v, attention_mask=None):
        # q, k, v: [B, H, L, D_head]
        # Transpose k for matmul: [B, H, D_head, L]
        w = torch.matmul(q, k.transpose(-2, -1)) 
        w = w / math.sqrt(v.size(-1))
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

        # --- RoPE Application ---
        if self.use_rope:
            # layer_past[0] shape is [B, H, L_past, D_head] (stored in logical order)
            past_length = 0 if layer_past is None else layer_past[0].size(-2)
            query_len = query.shape[2]
            
            # Get cos, sin
            cos, sin = self.rotary_emb(x=value, seq_len=past_length + query_len)
            
            # Apply RoPE to CURRENT query and key
            # Slice cos/sin for the current positions
            current_cos = cos[:, :, past_length : past_length + query_len, :]
            current_sin = sin[:, :, past_length : past_length + query_len, :]
            
            query, key = apply_rotary_pos_emb(query, key, current_cos, current_sin)

        # Handle Past (Caching)
        if layer_past is not None:
            # Assuming layer_past stores [key, value] in shape [B, H, L, D_head]
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # Save present for next step (Keep shape [B, H, L, D_head])
        present = (key, value) 

        # Attention Calculation
        a, attn_weights = self._attn(query, key, value, attention_mask) # [B, H, L, D_head]

        # Merge heads
        a = self._merge_heads(a, self.n_head, self.head_dim) # [B, L, D]
        a = self.c_proj(a)
        a = self.attn_dropout(a)

        # Output formatting matching original: [L, B, D], present, weights
        outputs = [a.transpose(0, 1), present, attn_weights]
        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, config, scale=False):
        # GPT2Configを作成し、元のconfigから属性をコピー
        gpt2config = GPT2Config(**config.__dict__)
        gpt2config.n_embd = config.embedding_dim
        # カスタムconfigにuse_ropeがある場合、gpt2configにも渡るように保証
        if hasattr(config, 'use_rope'):
            gpt2config.use_rope = config.use_rope
            gpt2config.rope_base = getattr(config, 'rope_base', 10000)

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
        self.use_rope = getattr(config, "use_rope", False)

        self.wte = nn.Embedding(config.vocab_size, nx)
        
        # RoPEを使う場合、絶対位置エンコーディング(PositionalEncoding)は通常不要です
        if self.use_rope:
            self.wpe = nn.Identity() # 何もしない層
        else:
            self.wpe = PositionalEncoding(nx, config.dropout, max_len=config.n_positions)
            
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        # (Memory pool関連は省略せずそのまま記述)
        self.ln_mem1 = nn.LayerNorm(nx)
        self.ln_mem2 = nn.LayerNorm(nx)
        self.ln_mem3 = nn.LayerNorm(nx)
        self.fc_latent = nn.Linear(3*nx, config.embedding_dim)

    def create_enc_attention_mask(self, input_ids):
        pad_array = input_ids == 0
        pad_array_ = pad_array.transpose(0, 1).unsqueeze(1).unsqueeze(2)
        return pad_array, torch.where(pad_array_ == True, float("-inf"), 0.0)
    
    def memory_pool(self, memory, pad_array):
        # 元のコードと同じ
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
        
        # wpeがIdentity(RoPE使用時)なら埋め込みそのまま、そうでなければ位置エンコーディング加算
        if self.use_rope:
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
        self.use_rope = getattr(config, "use_rope", False)

        self.wte = nn.Embedding(config.vocab_size, nx)
        
        # RoPE切り替え
        if self.use_rope:
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
            past = layer_past # Decoderのpastの渡し方を修正
            
        attention_mask = self.create_dec_attention_mask(x)
        input_embeds = self.wte(x)
        
        if self.use_rope:
            hidden_states = input_embeds
        else:
            hidden_states = self.wpe(input_embeds)
            
        # Latent injection
        hidden_states = hidden_states + latent.unsqueeze(1).transpose(0, 1)

        presents = ()
        for i, (block, layer_past_item) in enumerate(zip(self.h, past)):
            # Encoderとの違い: Decoderはautoregressiveなのでpastを正しく渡す
            outputs = block(hidden_states, layer_past=layer_past_item, attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
            presents = presents + (present,)
            
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.output_fc(hidden_states)
        # Return hidden_states and presents (for caching) if needed, but original only returned hidden
        return hidden_states 

# TransformerLatentなどの上位クラスはConfigに 'use_rope=True' を追加すればそのまま動作します。


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

        # Dropout の割合 (0以上なら適用)
        self.dropout_rate = config.dropout

        # バッチ正規化の有無 (Trueなら適用)
        self.use_batch_norm = config.batch_norm
        self.use_layer_norm = config.layer_norm

        # 各層のユニット数
        layer_dim = config.layer_dim
        layer_dim.insert(0, self.latent_dim)

        # Linear 層
        self.linear = nn.ModuleList([
            nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim)-1)
        ])

        # Batch Normalization 層 (フラグが True の場合のみ)
        if self.use_batch_norm:
            self.batch_norm = nn.ModuleList([
                nn.BatchNorm1d(layer_dim[i+1]) for i in range(len(layer_dim)-1)
            ])
        else:
            self.batch_norm = None

        # Layer Normalization 層 (フラグが True の場合のみ)

        if self.use_layer_norm:
            self.layer_norm = nn.ModuleList([
                nn.LayerNorm(layer_dim[i+1]) for i in range(len(layer_dim)-1)
            ])
        else:
            self.layer_norm = None


        # Dropout 層 (0 以上の値が設定されている場合のみ)
        if self.dropout_rate > 0:
            self.dropout = nn.ModuleList([
                nn.Dropout(self.dropout_rate) for _ in range(len(layer_dim)-1)
            ])
        else:
            self.dropout = None  # Dropout を適用しない場合は None

        # 最終分類層
        self.classifier = nn.Linear(layer_dim[-1], 1)

    def forward(self, x):
        for i, v in enumerate(self.linear):
            x = v(x)
            # if self.use_batch_norm and x.shape[0] > 1:  # バッチサイズが 1 のときは BatchNorm をスキップ
            #     x = self.batch_norm[i](x)
            # LayerNorm がある場合は適用
            # if self.use_layer_norm:
            #     x = self.layer_norm[i](x)
    
            x = self.activation(x)  # 活性化関数
            if self.dropout:  # Dropout が有効なら適用
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