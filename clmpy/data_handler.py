# -*- coding: utf-8 -*-
# 240316

from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import AllChem

from mordred import Calculator, descriptors

class BucketSampler(Sampler):
    def __init__(self,dataset,buckets=(20,150,10),shuffle=True,batch_size=512,drop_last=False):
        super().__init__(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        length = [len(v[0]) for v in dataset]

        bucket_range = np.arange(*buckets)
        
        assert isinstance(buckets,tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0
        buc = torch.bucketize(torch.tensor(length),torch.tensor(bucket_range),right=False)

        bucs = defaultdict(list)
        for i,v in enumerate(buc):
            bucs[v.item()].append(i)
        
        # remove empty bucket
        self.buckets = dict()
        for bucket_size, bucket in bucs.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket,dtype=torch.int)
        self.__iter__()

    def __iter__(self):
        # permutation in each bucket
        for bucket_size in self.buckets.keys():
            self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket,self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)
        # permutation of all batches
        if self.shuffle == True:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length


def tokenize(smiles,token_list):
    tokenized = []
    for s in smiles:
        s = s.replace("Br","R").replace("Cl","L")
        tok = []
        while len(s) > 0:
            if len(s) >= 2 and (s[0] == "@" or s[0] == "["):
                for j in np.arange(3,0,-1):
                    if s[:j] in token_list:
                        tok.append(s[:j])
                        s = s[j:]
                        break
            else:
                tok.append(s[0])
                s = s[1:]
        tokenized.append(tok)
    return tokenized

def sfl_tokenize(smiles,token_list):
    tokenized = []
    for s in smiles:
        s = s.replace("Br","R").replace("Cl","L")
        tok = []
        char = ""
        for v in s:
            if len(char) == 0 and v != "[":
                tok.append(v)
                continue
            char += v
            if len(char) > 1:
                if v == "]":
                    if char in token_list:
                        tok.append(char)
                    else:
                        tok.append("<unk>")
                    char = ""
        tokenized.append(tok)
    return tokenized
                
def one_hot_encoder(tokenized,token_dict):
    encoded = []
    for token in tokenized:
        enc = np.array([token_dict[v] for v in token])
        enc = np.concatenate([np.array([1]),enc,np.array([2])]).astype(np.int32)
        encoded.append(enc)
    return encoded

def seq2id(smiles,tokens,sfl=True):
    tok = sfl_tokenize if sfl else tokenize
    tokenized = tok(smiles,tokens.table)
    encoded = one_hot_encoder(tokenized,tokens.dict)
    return encoded

class tokens_table():
    def __init__(self,token_path):
        with open(token_path,"r") as f:
            tokens = f.read().replace("Br","R").replace("Cl","L").split("\n")
        self.table = tokens
        self.id2sm = {i:v for i,v in enumerate(tokens)}
        self.dict = {w:v for v,w in self.id2sm.items()}
        self.length = len(self.table)

class CLM_Dataset(Dataset):
    def __init__(self,x,y,token,sfl):
        self.tokens = token
        self.input = seq2id(x,self.tokens,sfl)
        self.output = seq2id(y,self.tokens,sfl)
        self.datanum = len(x)

    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        out_i = self.input[idx]
        out_o = self.output[idx]
        return out_i, out_o
    
class CLM_Dataset_MLP(Dataset):
    def __init__(self,x,y,binary,token,sfl):
        self.tokens = token
        self.input = seq2id(x,self.tokens,sfl)
        self.output = seq2id(y,self.tokens,sfl)
        self.output_y = torch.tensor(binary.iloc[:], dtype=torch.float)

        self.datanum = len(x)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_i = self.input[idx]
        out_o = self.output[idx]
        bin = self.output_y[idx]  # 修正: iloc を削除
        
        return out_i, out_o, bin


class SimpleDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = np.array(inputs, dtype=np.float32)  # 直接配列を受け取る
        self.labels = np.array(labels, dtype=np.float32)  # ラベルもNumPyに変換

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)  
        y = torch.tensor(self.labels[idx], dtype=torch.float32)  # ラベル
        return x, y


class CLM_Dataset_ECFP(Dataset):
    def __init__(self,x,y,binary,token,sfl):
        self.tokens = token
        self.input = torch.tensor(np.array(x.apply(smiles_to_mordred_fp).tolist()), dtype=torch.float32) 
        self.output = torch.tensor(np.array(x.apply(smiles_to_mordred_fp).tolist()), dtype=torch.float32)
        self.output_y = torch.tensor(binary.iloc[:], dtype=torch.float)

        self.datanum = len(x)
    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_i = self.input[idx]
        out_o = self.output[idx]
        bin = self.output_y[idx]  # 修正: iloc を削除
        
        return out_i, out_o, bin

def smiles_to_ecfp4(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)  # 無効なSMILESにはゼロベクトルを割り当て
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

def smiles_to_mordred_fp(smiles):
    calc = Calculator(descriptors, ignore_3D=False)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(calc.descriptors))  # 無効なSMILESにはゼロベクトルを返す

    # Mordredで記述子を計算
    desc = calc(mol)

    # 結果を辞書形式に変換 → numpy 配列に変換
    desc_values = np.array(list(desc.asdict().values()), dtype=np.float32)
    # NaN（計算できなかった値）をゼロに置き換える
    desc_values = np.nan_to_num(desc_values)

    return desc_values
   
class Encoder_Dataset(Dataset):
    def __init__(self,x,token,sfl):
        self.tokens = token
        self.input = seq2id(x,self.tokens,sfl)
        self.datanum = len(x)
    
    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        out_i = self.input[idx]
        return out_i

def collate(batch):
    xs, ys = [], []
    for x,y in batch:
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    ys = pad_sequence(ys,batch_first=False,padding_value=0)
    return xs, ys

def encoder_collate(batch):
    xs = []
    for x in batch:
        xs.append(torch.LongTensor(x))
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    return xs

def collate_MLP(batch):
    xs, ys, bins= [], [], []
    for x,y,bin in batch:
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
        bins.append(bin)
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    ys = pad_sequence(ys,batch_first=False,padding_value=0)
    bins = torch.tensor(bins)
    return xs, ys, bins

def collate_ECFP(batch):
    xs, ys, bins = [], [], []
    for x, y, bin in batch:
        xs.append(torch.FloatTensor(x))  # Long → Float に変更
        ys.append(torch.FloatTensor(y))  # Long → Float に変更
        bins.append(bin)  # ラベルは整数なので Long のままでOK

    xs = pad_sequence(xs, batch_first=True , padding_value=0)
    ys = pad_sequence(ys, batch_first=True , padding_value=0)
    bins = torch.tensor(bins, dtype=torch.long)  # ラベルは long でOK

    return xs, ys, bins
