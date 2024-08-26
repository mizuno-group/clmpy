# -*- coding: utf-8 -*-
# 240620

import os
from argparse import ArgumentParser, FileType
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import GRU
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
        self.epochs_run = 0
        self.ckpt_path = os.path.join(args.experiment_dir,"checkpoint.pt")
        if os.path.exists(self.ckpt_path):
            self._load(self.ckpt_path)
        self.best_model = None

    def _load(self,path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state.dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.steps_run = ckpt["step"]
        self.es.num_bad_epochs = ckpt["num_bad_epochs"]
        self.es.best = ckpt["es_best"]

    def _save(self,path,step):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
            "num_bad_ecochs": self.es.num_bad_epochs,
            "es_best": self.es.best
        }
        torch.save(ckpt,path)

    def _train_batch(self,source,target):
        self.optimizer.zero_grad()
        output, _ = self.model(source,target[:-1,:])
        l = self.criteria(output.transpose(-2,-1),target[1:,:])
        l.backward()
        self.optimizer.step()
        return l.item()
    
    def _valid_batch(self,source,target):
        output, _ = self.model(source,target[:-1,:])
        l = self.criteria(output.transpose(-2,-1),target[1:,:])
        return l.item()
    
    def train_epoch(self,epoch,device):
        self.model.train()
        l = []
        for source, target in self.train_data:
            source = source.to(device)
            target = target.to(device)
            l_ = self._train_batch(source,target)
            l.append(l_)
        return np.mean(l)
    
    def valid_epoch(self,device):
        self.model.eval()
        l = []
        with torch.no_grad():
            for source, target in self.valid_data:
                source = source.to(device)
                target = target.to(device)
                l_ = self._valid_batch(source,target)
                l.append(l_)
            l = np.mean(l)
            self.scheduler.step(l)
            end = self.es.step(l)
        return l, end
    
    def train(self,args):
        loss_t, loss_v = [], []
        for epoch in range(self.epochs_run,args.epochs):
            l = self.train_epoch(epoch,args.device)
            l_v, end = self.valid_epoch(args.device)
            if len(loss_v) == 0 or l_v < min(loss_v):
                self.best_model = self.model
            loss_t.append(l)
            loss_v.append(l_v)
            self._save(self.ckpt_path,epoch)
            print(f"epoch {epoch} | train_loss: {l}, valid_loss: {l_v}")
            if end:
                print(f"Early stopping at epoch {epoch}")
                return loss_t, loss_v
        return loss_t, loss_v
    

def main():
    args = get_args()
    print("loading data") 
    train_loader = prep_train_data(args)
    valid_loader = prep_valid_data(args)
    model = GRU(args)
    criteria, optimizer, scheduler, es = load_train_objs_gru(args,model)
    print("train start")
    trainer = Trainer(args,model,train_loader,valid_loader,criteria,optimizer,scheduler,es)
    loss_t, loss_v = trainer.train(args)
    torch.save(trainer.best_model.state_dict(),os.path.join(args.experiment_dir,"best_model.pt"))
    os.remove(trainer.ckpt_path)
    if args.plot:
        plot_loss(loss_t,loss_v,dir_name=args.experiment_dir)

if __name__ == "__main__":
    main()