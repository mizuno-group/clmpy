# -*- coding: utf-8 -*-
# 240316

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def KLLoss(mu,log_var):
    return 0.5 * (torch.sum(mu**2) + torch.sum(torch.exp(log_var)) - torch.sum(log_var) - log_var.numel()) / mu.shape[0]

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
        self.mu = nn.Linear(sum(self.enc_gru_layer),self.latent_dim)
        self.var = nn.Linear(sum(self.enc_gru_layer),self.latent_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        # x: Tensor, [L,B]
        embedding = self.embedding(x) # [L,B,E]
        _, states = self.gru(self.dropout(embedding))
        states = torch.cat([w(v) for v,w in zip(states,self.ln)],axis=1)
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
        return output, torch.cat(states,axis=1)

    def forward(self,x,state):
        # x: [L,B]
        # state: [B,H]
        hidden = self.linear(state)
        output, states = self.gru2out(x,hidden)
        return output, states


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
            if self.batch_norm and x.shape[0] > 1:  # バッチサイズが 1 のときは BatchNorm をスキップ
                x = self.batch_norm[i](x)
            x = self.activation(x)  # 活性化関数
            if self.dropout:  # Dropout が有効なら適用
                x = self.dropout[i](x)
        x = self.classifier(x)
        return x


class GRUVAE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder = Encoder(config)
        self.sampling = Sampling()
        self.decoder = Decoder(config)
    
    def forward(self,x,y):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu,log_var) # [B, H]
        out, hidden = self.decoder(y,z)
        return out, mu, log_var

class GRUVAE_MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder = Encoder(config)
        self.sampling = Sampling()
        self.decoder = Decoder(config)
        self.mlp = downstream_MLP(config)


    def forward(self,x,y):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu,log_var) # [B, H]
        out, hidden = self.decoder(y,z)
        out_d = self.mlp(mu)
        return out, out_d, mu, log_var



