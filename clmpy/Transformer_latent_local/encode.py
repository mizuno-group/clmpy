# -*- coding: utf-8 -*-
# 240513

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
    parser.add_argument("--config",type=FileType(mode="r"),default=None)
    parser.add_argument("--model_path",type=str,default="best_model.pt")
    parser.add_argument("--smiles_path", type=str, default="smiles.csv")  # CSVファイル想定
    parser.add_argument("--smiles_column", type=str, default="SMILES")   # 指定列名
    parser.add_argument("--save_dir",type=str, default=None)

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
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

def encode(args,smiles,model=None):
    if model == None:
        model = TransformerLatent(args).to(args.device)
        model.load_state_dict(torch.load(args.model_path))
    loader = prep_encode_data(args,smiles)        
    model.eval()
    res = []
    with torch.no_grad():
        for v in loader:
            latent = model.encoder(v.to(args.device))
            res.append(latent.cpu().detach().numpy())
    res = np.concatenate(res,axis=0)
    return res

def main():
    args = get_args()
    df = pd.read_csv(args.smiles_path)
    if args.smiles_column not in df.columns:
        raise ValueError(f"指定されたカラム '{args.smiles_column}' は {args.smiles_path} に存在しません。")
    
    smiles = df[args.smiles_column].dropna().astype(str).tolist()

    res = encode(args,smiles)
    pd.DataFrame(res,index=smiles).to_csv(os.path.join(args.save_dir,"encoded.csv"))


if __name__ == "__main__":
    main()