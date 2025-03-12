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
from ..model_helper import LossContainer
from ..utils import set_seed
from ..get_args import get_argument

class Trainer():
    def __init__(
        self,
        args,
        model: nn.Module,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        criteria: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        es,
    ):
        self.args = args
        self.model = model.to(args.device)
        self.train_data = train_data
        self.valid_data = prep_valid_data(args,valid_data)
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.es = es
        self.steps_run = 0
        self.loss = LossContainer()
        self.ckpt_path = os.path.join(args.experiment_dir,"checkpoint.pt")
        if os.path.exists(self.ckpt_path):
            self._load(self.ckpt_path)
        self.best_model = None
        self.device = args.device

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

    def _train_batch(self,source,target):
        self.model.train()
        self.optimizer.zero_grad()
        source = source.to(self.device)
        target = target.to(self.device)
        out, _ = self.model(source,target[:-1,:])
        l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
        l.backward()
        self.optimizer.step()
        self.scheduler.step()
        return l.item()
    
    def _valid_batch(self,source,target):
        self.model.eval()
        source = source.to(self.device)
        target = target.to(self.device)
        with torch.no_grad():
            out, _ = self.model(source,target[:-1,:])
            l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
        return l.item()
    
    def _train(self,train_data):
        min_l2 = float("inf")
        end = False
        for datas in train_data:
            self.steps_run += 1
            l_t = self._train_batch(*datas)
            if self.steps_run % self.args.valid_step_range == 0:
                l_v = []
                for v, w in self.valid_data:
                    l_v.append(self._valid_batch(v,w))
                l_v = np.mean(l_v)
                self.loss.train_add("reconstruction",l_t)
                self.loss.valid_add("reconstruction",l_v)
                end = self.es.step(l_v)
                if l_v < min_l2:
                    self.best_model = self.model
                    min_l2 = l_v
                self._save(self.ckpt_path,self.steps_run)
                if self.args.loss_log == True:
                    print(f"step {self.steps_run} | train_loss: {np.round(l_t,5)}, valid_loss: {np.round(l_v,5)}")
                if end:
                    print(f"Early stopping at step {self.steps_run}")
                    return end
            if self.steps_run >= self.args.steps:
                end = True
                return end
        return end
    
    def train(self):
        end = False
        while end == False:
            train_data = prep_train_data(self.args,self.train_data)
            end = self._train(train_data)
            if self.args.train_one_cycle == True:
                end = True
    

def main():
    args = get_argument()
    set_seed(args.seed)
    print("loading data")
    train_data = pd.read_csv(args.train_path,index_col=0)
    valid_data = pd.read_csv(args.valid_path,index_col=0) 
    model = GRU(args)
    criteria, optimizer, scheduler, es = load_train_objs(args,model)
    print("train start")
    trainer = Trainer(args,model,train_data,valid_data,criteria,optimizer,scheduler,es)
    trainer.train()
    torch.save(trainer.best_model.state_dict(),os.path.join(args.experiment_dir,"best_model.pt"))
    #if args.plot:
    #   plot_loss(loss_t,loss_v,dir_name=args.experiment_dir)

if __name__ == "__main__":
    ts = time.perf_counter()
    main()
    tg = time.perf_counter()
    dt = tg - ts
    h = dt // 3600
    m = (dt % 3600) // 60
    s = dt % 60
    print(f"elapsed time: {h} h {m} min {s} sec")