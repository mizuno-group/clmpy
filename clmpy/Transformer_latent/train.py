# -*- coding: utf-8 -*-
# 240527

import os
from argparse import ArgumentParser, FileType
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import TransformerLatent
from ..preprocess import *
from ..utils import plot_loss

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
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


class Trainer():
    def __init__(
        self,
        args,
        model: nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        criteria: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        es
    ):
        self.model = model.to(args.device)
        self.train_data = train_data
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

    def _save(self,path,step):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step
        }
        torch.save(ckpt,path)

    def _train_batch(self,source,target,device):
        self.model.train()
        self.optimizer.zero_grad()
        source = source.to(device)
        target = target.to(device)
        out, _ = self.model(source,target[:-1,:])
        l = self.criteria(out.transpose(-2,-1),target[1:,:])
        assert (not np.isnan(l.item()))
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
    
    def train(self,args):
        l, l2 = [], []
        min_l2 = float("inf")
        for step, datas in zip(range(self.steps_run,args.steps),self.train_data):
            l_t = self._train_batch(*datas,args.device)
            if step % args.valid_step_range == 0:
                l_v = 0
                for v, w in self.valid_data:
                    l_v += self._valid_batch(v,w,args.device)
                l.append(l_t)
                l2.append(l_v)
                end = self.es.step(l_v)
                if len(l) == 1 or l_v < min_l2:
                    self.best_model = self.model
                    min_l2 = l_v
                self._save(self.ckpt_path,step)
                print(f"step {step} | train_loss: {l_t}, valid_loss: {l_v}")
                if end:
                    print(f"Early stopping at step {step}")
                    return l, l2
        return l, l2
    
def main():
    args = get_args()
    print("loading data")
    train_loader = prep_train_data(args)
    valid_loader = prep_valid_data(args)
    model = TransformerLatent(args)
    criteria, optimizer, scheduler, es = load_train_objs_transformer(args,model)
    print("train start")
    trainer = Trainer(args,model,train_loader,valid_loader,criteria,optimizer,scheduler,es)
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