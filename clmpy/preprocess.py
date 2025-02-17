# -*- coding: utf-8 -*-
# 240516

import argparse
import yaml
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from .data_handler import *
from .utils import EarlyStopping, warmup_schedule


def load_train_objs(args,model,downstream=False):
    criteria = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.AdamW(model.parameters(),lr=args.max_lr)
    lr_schedule = warmup_schedule(args.warmup_step)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_schedule)
    es = EarlyStopping(patience=args.patience)
    if downstream == False:
        return criteria, optimizer, scheduler, es
    else:
        criteria_mlp = nn.BCEWithLogitsLoss()  
        return criteria, criteria_mlp, optimizer,scheduler, es 

def prep_train_data(args,train_data,downstream=False,bucketing=True):
    if downstream == True:
        trainset = CLM_Dataset(train_data["input"],train_data["output"],train_data["y"],args.token,args.SFL)
    else:
        trainset = CLM_Dataset(train_data["input"],train_data["output"],args.token,args.SFL)
    if bucketing == True:
        buckets = (args.buckets_min, args.buckets_max, args.buckets_step)
        train_sampler = BucketSampler(trainset,buckets,shuffle=args.batch_shuffle,batch_size=args.batch_size)
    else:
        train_sampler = BatchSampler(trainset,shuffle=args.batch_shuffle,batch_size=args.batch_size)
    train_loader = DataLoader(trainset,
                              batch_sampler=train_sampler,
                              collate_fn=collate,
                              num_workers=args.num_workers)
    return train_loader

def prep_valid_data(args,valid_data,downstream=False):
    if downstream == True:
        validset = CLM_Dataset(valid_data["input"],valid_data["output"],valid_data["y"],args.token,args.SFL)
    else:
        validset = CLM_Dataset(valid_data["input"],valid_data["output"],args.token,args.SFL)
    valid_loader = DataLoader(validset,
                              shuffle=False,
                              collate_fn=collate,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    return valid_loader

def prep_encode_data(args,smiles):
    dataset = Encoder_Dataset(smiles,args.token,args.SFL)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        collate_fn=encoder_collate,
                        num_workers=args.num_workers)
    return loader

def prep_token(token_path):
    tokens = tokens_table(token_path)
    return tokens
