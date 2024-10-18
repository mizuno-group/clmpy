import os
import re
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_loss(train,valid,train2=[],valid2=[],dir_name=""):
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(211)
    loss1 = ax1.plot(train,color="blue",label="train_rec")
    ax1.set_xlabel("step")
    ax1.set_ylabel("reconstruction loss")
    ax1.grid()
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    if len(train2) > 0:
        ax3 = ax1.twinx()
        loss3 = ax3.plot(train2,color="skyblue",label="train_KL")
        ax3.set_ylabel("KL loss")
        h3, l3 = ax3.get_legend_handles_labels()
        ax1.legend(h1+h3,l1+l3)

    ax2 = fig.add_subplot(212)
    loss2 = ax2.plot(valid,color="orange",label="valid_rec")
    ax2.set_xlabel("step")
    ax2.set_ylabel("reconstruction loss")
    ax2.grid()
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    h2, l2 = ax2.get_legend_handles_labels()
    if len(valid2) > 0:
        ax4 = ax2.twinx()
        loss4 = ax4.plot(valid2,color="yellow",label="valid_KL")
        ax4.set_ylabel("KL loss")
        h4, l4 = ax4.get_legend_handles_labels()
        ax2.legend(h2+h4,l2+l4)
    plt.savefig(os.path.join(dir_name,"loss.png"),bbox_inches="tight")


def generate_uniform_random(dim,n,low=-1,high=1):
    # dim: int, dimension of latent vector
    # n: int, sample size
    # low, high: int, range of random generation
    return np.array([np.random.uniform(low,high,dim) for i in range(n)]) # [n, d]

def generate_normal_random(n,mu,std):
    # n: sample size
    # mu, std: 1-d array or list
    return np.array([np.random.normal(loc,scale,n) for loc,scale in zip(mu,std)]).T #[n, d]

def sfl_token_list(smiles,out_path="tokens.txt"):
    lst = []
    for v in smiles:
        x = re.findall(r'\[[^\[]+\]',v)
        for y in x:
            lst.append(y)
    lst = list(set(lst))
    token = ['<pad>','<s>','</s>','0','1','2','3','4','5','6','7','8','9','(',')','=','#','@','*','%',
            '.','/','\\','+','-','c','n','o','s','p','H','B','C','N','O','P','S','F','Cl','Br','I']
    token.extend(sorted(lst))
    with open(out_path,"w") as f:
        f.write("\n".join(token))
    print("token length: {}".format(len(token)))


class EarlyStopping():
    def __init__(self,mode="min",min_delta=0,patience=10,percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_steps = 0
        self.is_better = None
        self._init_is_better(mode,min_delta,percentage)

        if patience == 0:
            self.is_better = lambda a,b: True
            self.step = lambda a: False

    def step(self,metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics,self.best):
            self.num_bad_steps = 0
            self.best = metrics
        else:
            self.num_bad_steps += 1

        if self.num_bad_steps >= self.patience:
            print("terminating because of early stopping.")
            return True
        
        return False

    def _init_is_better(self,mode,min_delta,percentage):
        if mode not in {"min","max"}:
            raise ValueError("mode "+mode+" is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best*min_delta/100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best*min_delta/100)

def warmup_schedule(warmup):
    def f(e):
        if e > 0:
            return min(e**-0.5,e*(warmup**-1.5))
        else:
            return 0
    return f


class attrdict(dict):
    def __init__(self,*args,**kwargs):
        dict.__init__(self,*args,**kwargs)
        self.__dict__ = self



