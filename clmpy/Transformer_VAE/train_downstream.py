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

from ..Transformer_latent.model import TransformerVAE_MLP, KLLoss
from ..preprocess import *
from ..utils import plot_loss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config",type=FileType(mode="r"),default=None)
    parser.add_argument("--model_path",type=str, default=None) #configに書くのもOK
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
        criteria_mlp : nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        es
    ):
        self.args = args
        self.model = model.to(args.device)
        self.train_data = prep_train_data(args,train_data,downstream=True)
        self.valid_data = prep_valid_data(args,valid_data,downstream=True)
        self.criteria = criteria
        self.criteria_mlp = criteria_mlp
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.es = es
        self.steps_run = 0
        self.ckpt_path = os.path.join(args.experiment_dir,"checkpoint.pt")
        if os.path.exists(self.ckpt_path):
            self._load_ckpt(self.ckpt_path)
        self.best_model = None
        self.device = args.device
        self.gamma = args.gamma
        self.total_step = args.steps
        self.valid_step_range = args.valid_step_range

    def _load_ckpt(self,path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.steps_run = ckpt["step"]
        self.es.num_bad_steps = ckpt["num_bad_steps"]
        self.es.best = ckpt["es_best"]

    def _load(self,path):
        self.model.load_state_dict(torch.load(path), strict=False)


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

    def _train_batch(self,source,target,target_mlp,device):
        self.model.train()
        self.optimizer.zero_grad()
        source = source.to(device)
        target = target.to(device)
        target_mlp = target_mlp.to(device)
        out, out_mlp, mu, log_var = self.model(source,target[:-1,:])
        target_mlp = target_mlp.float()
        l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
        l2 = KLLoss(mu,log_var) / source.shape[1]
        loss_mlp  = self.criteria_mlp(out_mlp,target_mlp.view(-1, 1))
        los = (l + l2 * self.beta) + self.gamma * loss_mlp
        assert (not np.isnan(l.item()))
        los.backward()
        self.optimizer.step()
        self.scheduler.step()
        return l.item(), l2.item(), loss_mlp.item()
    
    def _valid_batch(self,source,target,target_mlp,device):
        self.model.eval()
        source = source.to(device)
        target = target.to(device)
        target_mlp = target_mlp.to(device)
        target_mlp = target_mlp.float()
        with torch.no_grad():
            out, out_mlp, mu, log_var = self.model(source,target[:-1,:])
            l = self.criteria(out.transpose(-2,-1),target[1:,:]) / source.shape[1]
            loss_mlp  = self.criteria_mlp(out_mlp, target_mlp.view(-1, 1))
            l2 = KLLoss(mu,log_var) / source.shape[1]
        return l.item(), l2.item(), loss_mlp.item()

    
    def _train(self,train_data,log=True):
        lt, lv, lt2, lv2 , lm, lmv = [], [], [], [], [], []
        min_l2 = float("inf")
        end = False   
        for h, i, j in train_data:
            self.steps_run += 1
            l_t , l_t2, l_m = self._train_batch(h,i,j,self.device)
            if self.steps_run % self.valid_step_range == 0:
                l = []
                for v, w, y in self.valid_data:
                    l_v, l_v2, l_mv = self._valid_batch(v,w,y,self.device)
                    l.append(l_v + l_v2 * self.beta)
                l = np.mean(l_v)
                lt.append(l_t)
                lv.append(l_v)
                lm.append(l_m)
                lt2.append(l_t2)
                lv2.append(l_v2)
                lmv.append(l_mv)
                end = self.es.step(l_v)
                if len(l) == 1 or l_v < min_l2:
                    self.best_model = self.model
                    min_l2 = l_v
                self._save(self.ckpt_path,self.steps_run)
                if log:
                    print(f"step {self.steps_run} | train_loss: {(l_t + l_t2 * self.beta)+ l_m * self.gamma :.3f}, train_recon_loss:{l_t + l_t2 * self.beta:.3f}, train_mlp_loss:{l_m:.3f}, valid_loss: {l_v + l_v2 * self.beta:.3f}")
                if end:
                    print(f"Early stopping at step {self.steps_run}")
                    return lt, lv, lt2, lv2, end
            if self.steps_run >= self.total_step:
                end = True
                return lt, lv, lt2, lv2, end
        return lt, lv, lt2, lv2, end
    
    def train(self,log):
        end = False
        lt, lv, lt2, lv2 = [], [], [], []
        while end == False:
            l_t, l_v, l_t2, l_v2, end = self._train(self.train_data,log=log)
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
    model = TransformerVAE_MLP(args)
    criteria, criteria_mlp, optimizer, scheduler, es = load_train_objs_mlp(args,model)
    print("train start")
    trainer = Trainer(args,model,train_data,valid_data,criteria,criteria_mlp,optimizer,scheduler,es)
    if args.model_path is not None:
        trainer._load(args.model_path)
    loss_t, loss_v, loss_t2, loss_v2 = trainer.train(log=args.log)
    torch.save(trainer.best_model.state_dict(),os.path.join(args.experiment_dir,"best_model.pt"))
    os.remove(trainer.ckpt_path)
    """
    if args.plot:
        plot_loss(loss_t,loss_v,dir_name=args.experiment_dir)
    """

if __name__ == "__main__":
    ts = time.perf_counter()
    main()
    tg = time.perf_counter()
    dt = tg - ts
    h = dt // 3600
    m = (dt % 3600) // 60
    s = dt % 60
    print(f"elapsed time: {h} h {m} min {s} sec")