# -*- coding: utf-8 -*-
# 240516

import argparse
import yaml
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data_handler import *
from .utils import EarlyStopping, warmup_schedule

def load_train_objs_gru(args,model):
    criteria = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma)
    es = EarlyStopping(patience=args.patience)
    return criteria, optimizer, scheduler, es

def load_train_objs_transformer(args,model):
    criteria = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(),lr=args.max_lr)
    lr_schedule = warmup_schedule(args.warmup)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_schedule)
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

def prep_encode_data(args,smiles):
    dataset = Encoder_Dataset(smiles,args)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        collate_fn=encoder_collate,
                        num_workers=args.num_workers)
    return loader

def prep_token(args):
    tokens = tokens_table(args.token_path)
    return tokens

def get_notebook_args(config_file):
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    with open(config_file,"r") as f:
        config = yaml.safe_load(f)
    for v,w in config.items():
        args.__dict__[v] = w
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.token = prep_token(args)
    args.vocab_size = args.token.length
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args