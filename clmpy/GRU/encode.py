# -*- coding: utf-8 -*-
# 240620

import os
from argparse import ArgumentParser, FileType
import yaml

import numpy as np
import pandas as pd
import torch

from .model import GRU
from ..preprocess import prep_encode_data, prep_token
from ..get_args import get_argument

def encode(args,smiles,model):       
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
    args = get_argument()
    with open(args.smiles_path,"r") as f:
        smiles = f.read().split("\n")
    model = GRU(args).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    res = encode(args,smiles,model)
    output_path = os.path.join(args.experiment_dir,"encoded.csv") if len(args.output_path) == 0 else args.output_path
    pd.DataFrame(res,index=smiles).to_csv(output_path)

if __name__ == "__main__":
    main()