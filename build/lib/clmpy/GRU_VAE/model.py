# -*- coding: utf-8 -*-
# 240316

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def KLLoss(mu,log_var):
    return 0.5 * (torch.sum(mu**2) + torch.sum(torch.exp(log_var)) - torch.sum(log_var) - log_var.numel())

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
        return x, torch.cat(states,axis=1)


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
        self.mu = nn.Linear(sum(self.enc_gru_layer),self.latent_dim)
        self.var = nn.Linear(sum(self.enc_gru_layer),self.latent_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        # x: Tensor, [L,B]
        embedding = self.embedding(x) # [L,B,E]
        _, states = self.gru(self.dropout(embedding))
        mu = self.mu(states) # [B,H]
        log_var = self.var(states) # [B,H]
        return mu, log_var
    

class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,mu,log_var):
        epsilon = torch.randn(*mu.shape).to(DEVICE)
        return mu + torch.sqrt(torch.exp(log_var)) * epsilon
    

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
        return output, states

    def forward(self,x,state):
        # x: [L,B]
        # state: [B,H]
        hidden = self.linear(state)
        output, states = self.gru2out(x,hidden)
        return output, states
    

class VAE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder = Encoder(config)
        self.sampling = Sampling()
        self.decoder = Decoder(config)
    
    def forward(self,x,y):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu,log_var)
        out, hidden = self.decoder(y,z)
        return out, mu, log_var




