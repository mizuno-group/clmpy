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


def load_train_objs(args,model):
    criteria = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.AdamW(model.parameters(),lr=args.max_lr)
    lr_schedule = warmup_schedule(args.warmup)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_schedule)
    es = EarlyStopping(patience=args.patience)
    return criteria, optimizer, scheduler, es

def prep_train_data(args,train_data,downstream=False):
    buckets = (args.buckets_min, args.buckets_max, args.buckets_step)
    if downstream == True:
        trainset = CLM_Dataset_MLP(train_data["input"],train_data["output"],train_data["y"],args.token,args.SFL)
        cl = collate_MLP
    else:
        trainset = CLM_Dataset(train_data["input"],train_data["output"],args.token,args.SFL)
        cl = collate
    train_sampler = BucketSampler(trainset,buckets,shuffle=True,batch_size=args.batch_size)
    train_loader = DataLoader(trainset,
                              batch_sampler=train_sampler,
                              collate_fn=cl,
                              num_workers=args.num_workers)
    return train_loader

def prep_valid_data(args,valid_data,downstream=False):
    if downstream == True:
        validset = CLM_Dataset_MLP(valid_data["input"],valid_data["output"],valid_data["y"],args.token,args.SFL)
        cl = collate_MLP
    else:
        validset = CLM_Dataset(valid_data["input"],valid_data["output"],args.token,args.SFL)
        cl = collate
    valid_loader = DataLoader(validset,
                              shuffle=False,
                              collate_fn=cl,
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

def prep_token(token_path):
    tokens = tokens_table(token_path)
    return tokens

def get_notebook_args(config_file,**kwargs):
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    with open(config_file,"r") as f:
        config = yaml.safe_load(f)
    for v,w in config.items():
        args.__dict__[v] = w
    for v,w in kwargs:
        args.__dict__[v] = w
    try:
        args.patience = args.patience_step // args.valid_step_range
    except AttributeError:
        pass
    args.config = config_file
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.model_path = ""
    return args

# DataLoaderを作成
def get_dataloader(args, data):
    dataset = SimpleDataset(data.iloc[:, 4:].values, data["y"].values)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


def prep_train_data_ECFP(args,train_data):
    buckets = (args.buckets_min, args.buckets_max, args.buckets_step)
    trainset = CLM_Dataset_ECFP(train_data["input"],train_data["output"],train_data["y"],args.token,args.SFL) # メモリを抑えるオプションを入れたい
    train_sampler = BucketSampler(trainset,buckets,shuffle=True,batch_size=args.batch_size)
    train_loader = DataLoader(trainset,
                              batch_sampler=train_sampler,
                              collate_fn=collate_ECFP,
                              num_workers=args.num_workers)
    return train_loader

def prep_valid_data_ECFP(args,valid_data):
    validset = CLM_Dataset_ECFP(valid_data["input"],valid_data["output"],valid_data["y"],args.token,args.SFL)
    valid_loader = DataLoader(validset,
                              shuffle=False,
                              collate_fn=collate_ECFP,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    return valid_loader

class BrierScoreLoss(nn.Module):
    def __init__(self):
        super(BrierScoreLoss, self).__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)  # ロジットから確率に変換
        loss = torch.mean((probs - targets) ** 2)  # Brier Score の計算
        return loss

def load_train_objs_mlp(args,model,mode="min"):
    criteria = nn.CrossEntropyLoss(reduction="sum")
    if args.task == "binary_classification":
        #criteria_mlp = BrierScoreLoss()
        criteria_mlp = nn.BCEWithLogitsLoss()
        es = EarlyStopping(patience=args.patience,mode="max")
    elif args.task == "regression":
        criteria_mlp = nn.MSELoss()
        es = EarlyStopping(patience=args.patience,mode="min")
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    optimizer = optim.AdamW(model.parameters(),lr=args.max_lr)
    lr_schedule = warmup_schedule(args.warmup)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_schedule)
    
    return criteria, criteria_mlp, optimizer, scheduler, es