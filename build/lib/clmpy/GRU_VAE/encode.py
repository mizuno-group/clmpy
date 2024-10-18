# -*- coding: utf-8 -*-
# 240513

import os
from argparse import ArgumentParser, FileType
import yaml

import numpy as np
import pandas as pd
import torch

from .model import GRUVAE
from ..preprocess import prep_encode_data, prep_token


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config",type=FileType(mode="r"),default=None)
    parser.add_argument("--model_path",type=str,default="best_model.pt")
    parser.add_argument("--smiles_path",type=str,default="smiles.txt")
    args = parser.parse_args()
    config_dict = yaml.load(args.config,Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        arg_dict[key] = value
    args.config = args.config.name
    args.experiment_dir = "/".join(args.config.split("/")[:-1])
    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def encode(args,smiles,model=None):
    if model == None:
        model = GRUVAE(args).to(args.device)
        model.load_state_dict(torch.load(args.model_path))
    loader = prep_encode_data(args,smiles)       
    model.eval()
    res = []
    with torch.no_grad():
        for v in loader:
            mu, _ = model.encoder(v.to(args.device))
            res.append(mu.cpu().detach().numpy())
    res = np.concatenate(res,axis=0)
    return res
    
def main():
    args = get_args()
    with open(args.smiles_path,"r") as f: # smiles_path: txt of smiles list
        smiles = f.read().split("\n")
    res = encode(args,smiles)
    pd.DataFrame(res,index=smiles).to_csv(os.path.join(args.experiment_dir,"encoded.csv"))


if __name__ == "__main__":
    main()