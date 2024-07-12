# -*- coding: utf-8 -*-
# 240527

import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers.modeling_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    def __init__(self,config,scale=False):
        super().__init__(config)
        nx = config.embedding_dim
        self.n_head = config.n_head
        self.split_size = nx
        self.scale = scale
        self.head_dim = nx // self.n_head
        self.c_attn = Conv1D(3*nx,nx)
        self.c_proj = Conv1D(nx,nx)
        self.attn_dropout = nn.Dropout(config.dropout)

    def _attn(self,q,k,v,attention_mask=False):
        w = torch.matmul(q,k) # [B,H,L,L]
        w = w / math.sqrt(v.size(-1))
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)    
        outputs = torch.matmul(w,v) # [B,H,L,D/H]
        return outputs
    
    def forward(self,x,attention_mask=None,layer_past=None):
        # x: [L,B,D]
        x = self.c_attn(x).transpose(0,1) # [B,L,3D]
        query, key, value = x.split(self.split_size,dim=2) # [B,L,D] * 3
        query = self._split_heads(query,self.n_head,self.head_dim) # [B,H,L,D/H]
        key = self._split_heads(key,self.n_head,self.head_dim).transpose(-2,-1) # [B,H,D/H,L]
        value = self._split_heads(value,self.n_head,self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2,-1), layer_past[1]
            key = torch.cat((past_key,key),dim=-1)
            value = torch.cat((past_value,value),dim=-2)
        present = torch.stack((key.transpose(-2,-1),value)) # [B,L,2D]

        a = self._attn(query,key,value,attention_mask) # [B,H,L,D/H]
        a = self.attn_dropout(self.c_proj(self._merge_heads(a,self.n_head,self.head_dim)))
        outputs = [a.transpose(0,1),present]
        return outputs # [L,B,D]
    
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

    def create_enc_attention_mask(self,input_ids):
        pad_array = (input_ids == 0).transpose(0,1).unsqueeze(1).unsqueeze(2)
        return torch.where(pad_array == True, float("-inf"), 0.0) # [B,1,1,L]
    
    def memory_pool(self,memory,pad_array):
        mem = memory + torch.where(pad_array == True,float("-inf"),0).unsqueeze(2)
        mem = torch.where(mem == float("-inf"), None, mem)
        mx = torch.max(mem,dim=0)[0]
        ave = torch.mean(mem,dim=0)
        first = mem[0]
        return torch.cat([self.ln_mem1(mx),self.ln_mem2(ave),self.ln_mem3(first)],dim=1)
    
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

    def create_dec_attention_mask(self,input_ids):
        # input_ids: [L,B]
        l, b = input_ids.size()
        pad_array = (input_ids == 0).transpose(0,1).unsqueeze(1).unsqueeze(2) # [B,1,1,L]
        seq_array = torch.triu(torch.full((l,l),True,device=self.device),diagonal=1)
        seq_array = seq_array.unsqueeze(0).unsqueeze(1)
        res = torch.logical_or(pad_array,seq_array)
        return torch.where(res == True, float("-inf"), 0.0)
    
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
    


