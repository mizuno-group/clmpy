# -*- coding: utf-8 -*-
# 240318

import os
from argparse import ArgumentParser, FileType
import yaml
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import GRUVAE, KLLoss
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
    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.patience = args.patience_step // args.valid_step_range
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


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
        self.model = model.to(args.device)
        self.train_data = train_data
        self.valid_data = prep_valid_data(args,valid_data)
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.es = es
        self.beta = args.beta
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
        out, mu, log_var = self.model(source,target[:-1,:])
        l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
        l2 = KLLoss(mu,log_var) / source.shape[1]
        (l + l2 * self.beta).backward()
        self.optimizer.step()
        self.scheduler.step()
        return l.item(), l2.item()

    def _valid_batch(self,source,target,device):
        self.model.eval()
        source = source.to(device)
        target = target.to(device)
        with torch.no_grad():
            out, mu, log_var = self.model(source,target[:-1,:])
            l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
            l2 = KLLoss(mu,log_var) / source.shape[1]
        return l.item(), l2.item()

    def _train(self,args):
        lt, lv, lt2, lv2 = [], [], [], []
        min_l = float("inf")
        end = False
        for datas in self.train_data:
            self.steps_run += 1
            l_t, l_t2 = self._train_batch(*datas,args.device)
            if self.steps_run % args.valid_step_range == 0:
                l = []
                for v,w in self.valid_data:
                    l_v, l_v2 = self._valid_batch(v,w,args.device)
                    l.append(l_v + l_v2 * self.beta)
                l = np.mean(l)
                lt.append(l_t)
                lv.append(l_v)
                lt2.append(l_t2)
                lv2.append(l_v2)
                end = self.es.step(l)
                if l < min_l:
                    self.best_model = self.model
                    min_l = l
                self._save(self.ckpt_path,self.steps_run)
                print(f"step {self.steps_run} | train_loss: {l_t + l_t2 * self.beta}, valid_loss: {l}")
                if end:
                    print(f"Early stopping at step {self.steps_run}")
                    return lt, lv, lt2, lv2, end
            if self.steps_run >= args.steps:
                end = True
                return lt, lv, lt2, lv2, end
        return lt, lv, lt2, lv2, end

    def train(self,args):
        end = False
        lt, lv, lt2, lv2 = [], [], [], []
        while end == False:
            self.train_data = prep_train_data(args,self.train_data)
            l_t, l_v, l_t2, l_v2, end = self._train(args)
            lt.extend(l_t)
            lv.extend(l_v)
            lt2.extend(l_t2)
            lv2.extend(l_v2)
        return lt, lv, lt2, lv2
    

def main():
    args = get_args()
    set_seed(args.seed)
    print("loading data")
    train_data = pd.read_csv(args.train_data,index_col=0)
    valid_data = pd.read_csv(args.valid_data,index_col=0) 
    model = GRUVAE(args)
    criteria, optimizer, scheduler, es = load_train_objs(args,model)
    print("train start")
    trainer = Trainer(args,model,train_data,valid_data,criteria,optimizer,scheduler,es)
    loss_t, loss_v, loss_t2, loss_v2 = trainer.train(args)
    torch.save(trainer.best_model.state_dict(),os.path.join(args.experiment_dir,"best_model.pt"))
    os.remove(trainer.ckpt_path)
    if args.plot:
        plot_loss(loss_t,loss_v,loss_t2,loss_v2,dir_name=args.experiment_dir)


if __name__ == "__main__":
    ts = time.perf_counter()
    main()
    tg = time.perf_counter()
    dt = tg - ts
    h = dt // 3600
    m = (dt % 3600) // 60
    s = dt % 60
    print(f"elapsed time: {h} h {m} min {s} sec")
