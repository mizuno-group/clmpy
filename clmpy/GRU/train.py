# -*- coding: utf-8 -*-
# 240620

import os
from argparse import ArgumentParser, FileType
import yaml
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import GRU
from ..preprocess import *
from ..utils import plot_loss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config",type=FileType(mode="r"),default=None)
    args = parser.parse_args()
    config_dict = yaml.load(args.config,Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        arg_dict[key] = value
    args.config = args.config.name
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.token = prep_token(args)
    args.vocab_size = args.token.length
    args.patience = args.patience_step // args.valid_step_range
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


class Trainer():
    def __init__(
        self,
        args,
        model: nn.Module,
        valid_data: DataLoader,
        criteria: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        es,
    ):
        self.model = model.to(args.device)
        self.train_data = None
        self.valid_data = valid_data
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.es = es
        self.steps_run = 0
        self.ckpt_path = os.path.join(args.experiment_dir,"checkpoint.pt")
        if os.path.exists(self.ckpt_path):
            self._load(self.ckpt_path)
        self.best_model = None

    def _load(self,path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.steps_run = ckpt["step"]
        self.es.num_bad_steps = ckpt["num_bad_steps"]
        self.es.best = ckpt["es_best"]

    def _save(self,path,step):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
            "num_bad_steps": self.es.num_bad_steps,
            "es_best": self.es.best
        }
        torch.save(ckpt,path)

    def _train_batch(self,source,target,device):
        self.model.train()
        self.optimizer.zero_grad()
        source = source.to(device)
        target = target.to(device)
        out, _ = self.model(source,target[:-1,:])
        l = self.criteria(out.transpose(-2,-1),target[1:,:])
        l.backward()
        self.optimizer.step()
        self.scheduler.step()
        return l.item()
    
    def _valid_batch(self,source,target,device):
        self.model.eval()
        source = source.to(device)
        target = target.to(device)
        with torch.no_grad():
            out, _ = self.model(source,target[:-1,:])
            l = self.criteria(out.transpose(-2,-1),target[1:,:])
        return l.item()
    
    def _train(self,args):
        l, l2 = [], []
        min_l2 = float("inf")
        end = False
        for datas in self.train_data:
            self.steps_run += 1
            l_t = self._train_batch(*datas,args.device)
            if self.steps_run % args.valid_step_range == 0:
                l_v = []
                for v, w in self.valid_data:
                    l_v.append(self._valid_batch(v,w,args.device))
                l_v = np.mean(l_v)
                l.append(l_t)
                l2.append(l_v)
                end = self.es.step(l_v)
                if len(l) == 1 or l_v < min_l2:
                    self.best_model = self.model
                    min_l2 = l_v
                self._save(self.ckpt_path,self.steps_run)
                print(f"step {self.steps_run} | train_loss: {l_t}, valid_loss: {l_v}")
                if end:
                    print(f"Early stopping at step {self.steps_run}")
                    return l, l2, end
            if self.steps_run >= args.steps:
                end = True
                return l, l2, end
        return l, l2, end
    
    def train(self,args):
        end = False
        l, l2 = [], []
        while end == False:
            self.train_data = prep_train_data(args)
            a, b, end = self._train(args)
            l.extend(a)
            l2.extend(b)
        return l, l2
    

def main():
    args = get_args()
    set_seed(args.seed)
    print("loading data") 
    valid_loader = prep_valid_data(args)
    model = GRU(args)
    criteria, optimizer, scheduler, es = load_train_objs(args,model)
    print("train start")
    trainer = Trainer(args,model,valid_loader,criteria,optimizer,scheduler,es)
    loss_t, loss_v = trainer.train(args)
    torch.save(trainer.best_model.state_dict(),os.path.join(args.experiment_dir,"best_model.pt"))
    os.remove(trainer.ckpt_path)
    if args.plot:
        plot_loss(loss_t,loss_v,dir_name=args.experiment_dir)

if __name__ == "__main__":
    ts = time.perf_counter()
    main()
    tg = time.perf_counter()
    dt = tg - ts
    h = dt // 3600
    m = (dt % 3600) // 60
    s = dt % 60
    print(f"elapsed time: {h} h {m} min {s} sec")