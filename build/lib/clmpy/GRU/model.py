# -*- coding: utf-8 -*-
# 240620

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class GRU_Layer(nn.Module):
    def __init__(self,embedding_dim,layer):
        super().__init__()
        self.layer = layer
        self.embedding_dim = embedding_dim
        dims = self.layer.copy()
        dims.insert(0,self.embedding_dim)

        self.gru = nn.ModuleList([nn.GRU(dims[i],dims[i+1],1) for i in range(len(self.layer))])

    def forward(self,x,h=None):
        # x: [L,B,E]
        # h: list of [B,Hn]
        states = []
        if h == None:
            for v in self.gru:
                x, s = v(x)
                states.append(s.squeeze(0))
        else:
            for v, state in zip(self.gru,h):
                state = state.unsqueeze(0).contiguous()
                x, s = v(x,state) 
                states.append(s.squeeze(0))
        return x, states
    

class Encoder(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        enc_gru_layer: list of int, the size of GRU hidden units
        latent_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.enc_gru_layer = config.enc_gru_layer
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.latent_dim = config.latent_dim

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=0)
        self.gru = GRU_Layer(self.embedding_dim,self.enc_gru_layer)
        self.ln = nn.ModuleList([nn.LayerNorm(v) for v in self.enc_gru_layer])
        self.linear = nn.Linear(sum(self.enc_gru_layer),self.latent_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x,inference=False):
        # x: Tensor, [L,B]
        embedding = self.embedding(x) # [L,B,E]
        _, states = self.gru(self.dropout(embedding))
        states = torch.cat([w(v) for v,w in zip(states,self.ln)],axis=1)
        latent = self.linear(states)
        if inference == False:
            latent += torch.normal(0,0.05,size=latent.shape).to(DEVICE)
        return torch.tanh(latent)


class Decoder(nn.Module):
    def __init__(self,config):
        """
        vocab_size: int, the number of input words
        embedding_dim: int, embedding dimention
        dec_gru_layer: list of int, the size of GRU hidden units
        latent_dim: int, the unit size of bottleneck layer
        dropout: float [0,1], Dropout ratio
        """
        super().__init__()
        self.dec_gru_layer = config.dec_gru_layer
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.latent_dim = config.latent_dim
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim,padding_idx=0)
        self.linear = nn.Linear(self.latent_dim,sum(self.dec_gru_layer))
        self.gru = GRU_Layer(self.embedding_dim,self.dec_gru_layer)
        self.linear_out = nn.Linear(self.dec_gru_layer[-1],self.vocab_size,bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def gru2out(self,x,hidden):
        embedding = self.embedding(x)
        hidden = torch.split(hidden,self.dec_gru_layer,dim=1)
        embedding, states = self.gru(self.dropout(embedding),hidden)
        output = self.linear_out(embedding)
        return output, torch.cat(states,axis=1)

    def forward(self,x,state):
        # x: [L,B]
        # state: [B,H]
        hidden = self.linear(state)
        output, states = self.gru2out(x,hidden)
        return output, states
    

class GRU(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self,x,y):
        latent = self.encoder(x)
        out, hidden = self.decoder(y,latent)
        return out, latent