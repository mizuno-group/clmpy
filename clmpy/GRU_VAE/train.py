# -*- coding: utf-8 -*-
# 240318

import os
from argparse import ArgumentParser, FileType
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import GRUVAE, KLLoss
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
        es,
    ):
        self.model = model.to(args.device)
        self.train_data = train_data
        self.valid_data = valid_data
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.es = es
        self.beta = args.beta
        self.epochs_run = 0
        self.ckpt_path = os.path.join(args.experiment_dir,"checkpoint.pt")
        if os.path.exists(self.ckpt_path):
            self._load(self.ckpt_path)
        self.best_model = None

    def _load(self,path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epochs_run = ckpt["epoch"]

    def _save(self,path,epoch):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt,path)

    def _train_batch(self,source,target):
        self.optimizer.zero_grad()
        output, mu, log_var = self.model(source,target[:-1,:])
        l = self.criteria(output.transpose(-2,-1),target[1:,:])
        l2 = KLLoss(mu,log_var)
        (l + l2 * self.beta).backward()
        self.optimizer.step()
        return l.item(), l2.item()
    
    def _valid_batch(self,source,target):
        output, mu, log_var = self.model(source,target[:-1,:])
        l = self.criteria(output.transpose(-2,-1),target[1:,:])
        l2 = KLLoss(mu,log_var)
        return l.item(), l2.item()
    
    def train_epoch(self,epoch,device):
        self.model.train()
        l, l2 = [], []
        for source, target in self.train_data:
            source = source.to(device)
            target = target.to(device)
            l_ = self._train_batch(source,target)
            l.append(l_[0])
            l2.append(l_[1])
        return np.mean(l), np.mean(l2)
    
    def valid_epoch(self,device):
        self.model.eval()
        l, l2 = [], []
        with torch.no_grad():
            for source, target in self.valid_data:
                source = source.to(device)
                target = target.to(device)
                l_ = self._valid_batch(source,target)
                l.append(l_[0])
                l2.append(l_[1])
            l = np.mean(l)
            l2 = np.mean(l2)
            self.scheduler.step()
            end = self.es.step(l+l2*self.beta)
        return l, l2, end
    
    def train(self,args):
        loss_t, loss_t2, loss_v, loss_v2 = [], [], [], []
        for epoch in range(self.epochs_run,args.epochs):
            l, l2 = self.train_epoch(epoch,args.device)
            l_v, l_v2, end = self.valid_epoch(args.device)
            if len(loss_v) == 0 or l_v < min(loss_v):
                self.best_model = self.model
            loss_t.append(l)
            loss_t2.append(l2)
            loss_v.append(l_v)
            loss_v2.append(l_v2)
            self._save(self.ckpt_path,epoch)
            print(f"epoch {epoch} | train_loss: {l+l2*self.beta}, valid_loss: {l_v+l_v2*self.beta}")
            if end:
                print(f"Early stopping at epoch {epoch}")
                return loss_t, loss_t2, loss_v, loss_v2
        return loss_t, loss_t2, loss_v, loss_v2
            

def main():
    args = get_args()
    print("loading data") 
    train_loader = prep_train_data(args)
    valid_loader = prep_valid_data(args)
    model = GRUVAE(args)
    criteria, optimizer, scheduler, es = load_train_objs_gru(args,model)
    print("train start")
    trainer = Trainer(args,model,train_loader,valid_loader,criteria,optimizer,scheduler,es)
    loss_t, loss_t2, loss_v, loss_v2 = trainer.train(args)
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
