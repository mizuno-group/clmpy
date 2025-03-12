from collections import defaultdict
import torch

class LossContainer():
    def __init__(self):
        self.train_loss = defaultdict(list)
        self.valid_loss = defaultdict(list)

    def train_add(self,key,value):
        self.train_loss[key].append(value)

    def valid_add(self,key,value):
        self.valid_loss[key].append(value)


def KLLoss(mu,log_var):
    return 0.5 * (torch.sum(mu**2) + torch.sum(torch.exp(log_var)) - torch.sum(log_var) - log_var.numel()) / mu.shape[0]