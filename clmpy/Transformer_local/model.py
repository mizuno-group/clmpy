# -*- coding: utf-8 -*-
# 240527

import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

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

    # --- Transformers 4.x 対応: split_heads / merge_heads を自前実装 ---
    def _split_heads(self, x, num_heads, head_dim):
        # x: [B, L, D]
        B, L, D = x.size()
        x = x.view(B, L, num_heads, head_dim)   # [B, L, H, D/H]
        return x.permute(0, 2, 1, 3)            # [B, H, L, D/H]

    def _merge_heads(self, x, num_heads, head_dim):
        # x: [B, H, L, D/H]
        B, H, L, d = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, H, D/H]
        return x.view(B, L, H * d)              # [B, L, D]
    # --------------------------------------------------------------------

    def _attn(self, q, k, v, attention_mask=False):
        w = torch.matmul(q, k)  # [B,H,L,L]
        w = w / math.sqrt(v.size(-1))
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        outputs = torch.matmul(w, v)  # [B,H,L,D/H]
        return outputs

    def forward(self, x, attention_mask=None, layer_past=None):
        # x: [L,B,D]
        x = self.c_attn(x).transpose(0, 1)  # [B,L,3D]
        query, key, value = x.split(self.split_size, dim=2)  # [B,L,D] * 3

        query = self._split_heads(query, self.n_head, self.head_dim)           # [B,H,L,D/H]
        key = self._split_heads(key, self.n_head, self.head_dim).transpose(-2, -1)  # [B,H,D/H,L]
        value = self._split_heads(value, self.n_head, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value))  # [B,L,2D]

        a = self._attn(query, key, value, attention_mask)  # [B,H,L,D/H]
        a = self.attn_dropout(self.c_proj(self._merge_heads(a, self.n_head, self.head_dim)))

        outputs = [a.transpose(0, 1), present]
        return outputs  # [L,B,D]

class TransformerBlock(nn.Module):
    def __init__(self,config,scale=False):
        gpt2config = GPT2Config(**config.__dict__)
        gpt2config.n_embd = config.embedding_dim
        super().__init__()
        nx = config.embedding_dim
        self.ln_1 = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.attn = Attention(gpt2config,scale)
        self.ln_2 = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4*nx,gpt2config)
    
    def forward(self,x,attention_mask=None,layer_past=None):
        # x: [L,B,D]
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
        self.window_size = config.window_size

    
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
        L, B = input_ids.size()
        device = input_ids.device
        
        # 1. Padding Mask (既存の処理)
        # input_ids == 0 の場所をTrueにする
        pad_array = input_ids == 0  # [L, B]
        # Attentionの形状 [B, 1, 1, L] に合わせる (Target Lengthへのマスク)
        pad_mask = pad_array.transpose(0, 1).unsqueeze(1).unsqueeze(2) 

        # 2. Sliding Window Mask (新規追加)
        # 行列の各要素 (i, j) について、|i - j| > window_size ならマスクする
        window_size = self.window_size
        
        sliding_mask = None
        if window_size > 0:
            # [L, 1] - [1, L] で (i - j) の行列を作る
            indices = torch.arange(L, device=device)
            diff = indices.unsqueeze(1) - indices.unsqueeze(0)
            # 距離が window_size を超えている場所を True
            sliding_mask = torch.abs(diff) > window_size
            # 形状を [1, 1, L, L] にしてブロードキャスト可能にする
            sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)

        # 3. マスクの結合
        # Padding または SlidingWindow の範囲外であれば True
        if sliding_mask is not None:
            # logical_or で結合
            final_mask = torch.logical_or(pad_mask, sliding_mask)
        else:
            final_mask = pad_mask

        # True の場所を -inf に、それ以外を 0.0 に
        return pad_array, torch.where(final_mask, -10000.0, 0.0)

    def forward(self,x,past=None):
        # x: Tensor, [L,B]
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
        self.window_size = config.window_size
       
    def create_dec_attention_mask(self, input_ids):
        # input_ids: [L, B]
        L, B = input_ids.size()
        device = input_ids.device # self.device よりも input_ids.device を使うほうが安全

        # 1. Padding Mask
        pad_array = (input_ids == 0).transpose(0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]

        # 2. Causal Mask (未来を見ない)
        # triu(diagonal=1) で対角線より上（未来）をTrueにする
        causal_mask = torch.triu(torch.full((L, L), True, device=device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # [1, 1, L, L]

        # 3. Sliding Window Mask (過去を見すぎない) (新規追加)
        window_size = self.window_size
        
        sliding_mask = None
        if window_size > 0:
            # indices[i] - indices[j] > window_size  => 過去すぎる
            indices = torch.arange(L, device=device)
            diff = indices.unsqueeze(1) - indices.unsqueeze(0) # diff[i, j] = i - j
            # i - j > window_size の場所を True (過去方向の制限のみ)
            # ※ 未来方向は causal_mask で消えるので abs ではなく単なる差でOKだが、absでも問題はない
            sliding_mask = diff > window_size 
            sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(1) # [1, 1, L, L]

        # 4. マスクの結合
        # 元の pad + causal
        res = torch.logical_or(pad_array, causal_mask)
        
        # さらに sliding window を追加
        if sliding_mask is not None:
            res = torch.logical_or(res, sliding_mask)

        return torch.where(res, -10000.0, 0.0)

    def forward(self,x,latent,layer_past=None):
        # x: [L,B]
        # latent: [B,D]
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
        return hidden_states # [L,B,V]
    

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