import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

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

class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,dropout,max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len,embedding_dim)
        position = torch.arange(0,max_len).unsqueeze(1) #[maxlen, 1]
        div_term = torch.exp(torch.arange(0,embedding_dim,2) *
                             -(math.log(10000.0) / embedding_dim))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe",pe)

    def forward(self,x):
        # x: [L,B,D]
        x = x + Variable(self.pe[:x.size(0)],
                         requires_grad=False)
        return self.dropout(x)

# --- Role-Based Attention に変更 ---
class RoleBasedAttention(GPT2Attention):
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
        
        # ★ 追加: ウィンドウサイズとローカルヘッドの割合設定
        # config に window_size が無い場合のデフォルト値も設定しておくと安全
        self.window_size = config.window_size
        # 半分のヘッドをローカル担当にする（必要に応じて config から取得するように変更可）
        self.n_local_heads = config.n_local_heads

    def _split_heads(self, x, num_heads, head_dim):
        B, L, D = x.size()
        x = x.view(B, L, num_heads, head_dim)
        return x.permute(0, 2, 1, 3) # [B, H, L, D/H]

    def _merge_heads(self, x, num_heads, head_dim):
        B, H, L, d = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, H * d)

    def _create_local_mask(self, length, device):
        """ウィンドウサイズ外を隠すマスクを作成 (対角成分周辺のみ通す)"""
        # 全て True (マスクする) で初期化
        mask = torch.ones((length, length), device=device).bool()
        # バンド行列（ウィンドウ内）を False (マスクしない) にする
        # triu/tril で対角線からの距離を指定
        window_mask = torch.triu(torch.ones_like(mask), diagonal=-self.window_size) * \
                      torch.tril(torch.ones_like(mask), diagonal=self.window_size)
        
        # window_mask が 1 の場所は通す(False)、0 の場所は隠す(True)
        return (window_mask == 0)

    def _attn(self, q, k, v, attention_mask=None):
        # q, k, v: [B, H, L, D/H]
        w = torch.matmul(q, k)
        w = w / math.sqrt(v.size(-1)) # [B, H, L, L]

        # --- ★ ここでヘッドごとのマスク処理を行う ---
        B, H, L, _ = w.size()
        
        # 1. ベースとなるマスク（Padding や Causal）を適用
        # attention_mask は [B, 1, 1, L] や [B, 1, L, L] で来る想定
        if attention_mask is not None:
            # ブロードキャストで加算 [B, H, L, L]
            w = w + attention_mask

        # 2. ローカルヘッドにのみ「ウィンドウ制限」を追加適用
        if self.n_local_heads > 0 and self.window_size > 0:
            # ローカルマスク作成 [L, L]
            local_mask_bool = self._create_local_mask(L, w.device)
            # 形状合わせ [1, 1, L, L]
            local_mask_val = torch.zeros_like(local_mask_bool, dtype=w.dtype).masked_fill_(local_mask_bool, -10000.0)
            local_mask_val = local_mask_val.unsqueeze(0).unsqueeze(0)
            
            # 該当するヘッドにのみ加算 (inplace operationによるエラー回避のため clone を推奨する場合もあるがここでは直接加算)
            # w[:, :self.n_local_heads, :, :] += local_mask_val 
            # 安全策: 分割して結合
            w_local = w[:, :self.n_local_heads, :, :] + local_mask_val
            w_global = w[:, self.n_local_heads:, :, :]
            w = torch.cat([w_local, w_global], dim=1)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        outputs = torch.matmul(w, v)
        return outputs

    def forward(self, x, attention_mask=None, layer_past=None):
        x = self.c_attn(x).transpose(0, 1)
        query, key, value = x.split(self.split_size, dim=2)

        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim).transpose(-2, -1)
        value = self._split_heads(value, self.n_head, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value))
        
        # attention_mask は上位(Encoder/Decoder)から渡される「共通マスク」
        a = self._attn(query, key, value, attention_mask)
        
        a = self.attn_dropout(self.c_proj(self._merge_heads(a, self.n_head, self.head_dim)))
        outputs = [a.transpose(0, 1), present]
        return outputs

class TransformerBlock(nn.Module):
    def __init__(self,config,scale=False):
        gpt2config = GPT2Config(**config.__dict__)
        gpt2config.n_embd = config.embedding_dim
        super().__init__()
        nx = config.embedding_dim
        self.ln_1 = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        # ★ RoleBasedAttention に変更
        self.attn = RoleBasedAttention(gpt2config,scale)
        self.ln_2 = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4*nx,gpt2config)
    
    def forward(self,x,attention_mask=None,layer_past=None):
        output_attn = self.attn(self.ln_1(x),attention_mask,layer_past)
        a = output_attn[0]
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = [x] + output_attn[1:]
        return outputs
    

class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        nx = config.embedding_dim
        self.wte = nn.Embedding(config.vocab_size,nx)
        self.wpe = PositionalEncoding(nx,config.dropout,max_len=config.n_positions)
        self.drop = nn.Dropout(config.dropout)

        self.h = nn.ModuleList([TransformerBlock(config,scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.ln_mem1 = nn.LayerNorm(nx)
        self.ln_mem2 = nn.LayerNorm(nx)
        self.ln_mem3 = nn.LayerNorm(nx)
        self.fc_latent = nn.Linear(3*nx,config.embedding_dim)
        # self.window_size = config.window_size # Attention内で処理するためここでは不要だが保持しても良い

    def memory_pool(self,memory,pad_array):
        pad_array = pad_array.unsqueeze(-1)
        masked = memory.masked_fill(pad_array,-torch.inf)
        padding_mask = ~pad_array
        mx = torch.max(masked,dim=0)[0]
        ave = torch.sum(memory*padding_mask,dim=0) / torch.sum(padding_mask,dim=0)
        first = memory[0]
        return torch.cat([self.ln_mem1(mx),self.ln_mem2(ave),self.ln_mem3(first)],dim=1)

    def create_enc_attention_mask(self, input_ids):
        # input_ids: [L, B]
        # ★ ここでの Sliding Window ロジックは削除し、純粋な Padding Mask のみに戻す ★
        # これにより、RoleBasedAttention 内で Global ヘッドは全体を見れるようになる
        
        pad_array = input_ids == 0
        pad_mask = pad_array.transpose(0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]
        
        return pad_array, torch.where(pad_mask, -10000.0, 0.0)

    def forward(self,x,past=None):
        input_shape = x.size()
        x = x.view(-1,input_shape[-1])
        if past is None:
            past = [None] * len(self.h)
        input_embeds = self.wte(x)
        hidden_states = self.wpe(input_embeds)
        pad_array, attention_mask = self.create_enc_attention_mask(x)
        output_shape = input_shape + (hidden_states.size(-1),)

        for i, (block, layer_past) in enumerate(zip(self.h,past)):
            outputs = block(hidden_states,layer_past=layer_past,attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        latent = self.memory_pool(hidden_states,pad_array)
        latent = self.fc_latent(latent)
        return torch.tanh(latent) 
    

class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        nx = config.embedding_dim
        self.wte = nn.Embedding(config.vocab_size,nx)
        self.wpe = PositionalEncoding(nx,config.dropout,max_len=config.n_positions)
        self.input_proj = nn.Linear(nx,nx,bias=False)
        self.h = nn.ModuleList([TransformerBlock(config,scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.output_fc = nn.Linear(nx,config.vocab_size)
        self.device = config.device
        # self.window_size = config.window_size # Attention内で処理

    def create_dec_attention_mask(self, input_ids):
        # input_ids: [L, B]
        L, B = input_ids.size()
        device = input_ids.device

        # 1. Padding Mask
        pad_array = (input_ids == 0).transpose(0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]

        # 2. Causal Mask (未来を見ない)
        causal_mask = torch.triu(torch.full((L, L), True, device=device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # [1, 1, L, L]

        # ★ ここでの Sliding Window ロジックも削除 ★
        # RoleBasedAttention に任せることで、Global ヘッドは過去全てを参照可能にする
        
        res = torch.logical_or(pad_array, causal_mask)
        return torch.where(res, -10000.0, 0.0)
    
    def forward(self,x,latent,layer_past=None):
        if layer_past is None:
            past = [None] * len(self.h)
        attention_mask = self.create_dec_attention_mask(x)
        input_embeds = self.wte(x)
        hidden_states = self.wpe(input_embeds)
        hidden_states = hidden_states + latent.unsqueeze(1).transpose(0,1)

        presents = ()
        for i, (block, layer_past) in enumerate(zip(self.h,past)):
            outputs = block(hidden_states,layer_past=layer_past,attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
            presents = presents + (present,)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.output_fc(hidden_states)
        return hidden_states

# --- 以下のクラスは変更なし ---

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
            # BN/LN/Dropoutの処理 (省略されていた部分を補完)
            if self.batch_norm is not None and x.shape[0] > 1:
                x = self.batch_norm[i](x)
            if self.layer_norm is not None:
                x = self.layer_norm[i](x)
            
            x = self.activation(x)
            
            if self.dropout is not None:
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