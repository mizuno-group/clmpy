# -*- coding: utf-8 -*-
# 240603

import os
from argparse import ArgumentParser, FileType
import yaml

import numpy as np
import pandas as pd
import torch

from .model import TransformerLatent
from ..preprocess import *

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config",type=FileType(mode="r"),default="config.yml")
    parser.add_argument("--model_path",type=str,default="best_model.pt")
    parser.add_argument("--test_path",type=str,default="data/val_10k.csv")
    args = parser.parse_args()
    config_dict = yaml.load(args.config,Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        arg_dict[key] = value
    args.config = args.config.name
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.token = prep_token(args)
    args.vocab_size = args.token.length
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


class Evaluator():
    def __init__(self,model,args):
        self.id2sm = args.token.id2sm
        self.model = model.to(args.device)
        self.maxlen = args.maxlen
        self._load(args.model_path)

    def _load(self,path):
        self.model.load_state_dict(torch.load(path))

    def _eval_batch(self,source,target,device):
        source = source.to(device)
        latent, _ = self.model.encoder(source)
