# -*- coding: utf-8 -*-
# 240516

import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_handler import *
from utils import EarlyStopping

def load_train_objs(args,model):
    criteria = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma)
    es = EarlyStopping(patience=args.patience)
    return criteria, optimizer, scheduler, es

def prep_train_data(args):
    buckets = (args.buckets_min, args.buckets_max, args.buckets_step)
    train = pd.read_csv(args.train_data,index_col=0)
    trainset = CLM_Dataset(train["random"],train["canonical"],args)
    train_sampler = BucketSampler(trainset,buckets,shuffle=True,batch_size=args.batch_size)
    train_loader = DataLoader(trainset,
                              batch_sampler=train_sampler,
                              collate_fn=collate,
                              num_workers=args.num_workers)
    return train_loader

def prep_valid_data(args):
    valid = pd.read_csv(args.valid_data,index_col=0)
    validset = CLM_Dataset(valid["random"],valid["canonical"],args)
    valid_loader = DataLoader(validset,
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    return valid_loader

def prep_encode_data(args):
    with open(args.smiles_path,"r") as f:
        smiles = f.read().split("\n")
    dataset = Encoder_Dataset(smiles,args)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        collate_fn=encoder_collate,
                        num_workers=args.num_workers)
    return loader