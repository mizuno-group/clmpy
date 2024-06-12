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
        self.maxlen = args.n_positions
        self._load(args.model_path)

    def _load(self,path):
        self.model.load_state_dict(torch.load(path))

    def _eval_batch(self,source,target,device):
        source = source.to(device)
        latent = self.model.encoder(source)
        token_ids = np.zeros((self.maxlen,source.size(1)))
        token_ids[0:,] = 1
        token_ids = torch.tensor(token_ids,dtype=torch.long).to(device)
        for i in range(1,self.maxlen):
            token_ids_seq = token_ids[:i,:]
            out = self.model.decoder(token_ids_seq,latent)
            _, out_id = out.max(dim=2)
            new_id = out_id[-1,:]
            is_end_token = token_ids[i-1,:] == 0
            is_pad_token = token_ids[i-1,:] == 2
            judge = torch.logical_or(is_end_token,is_pad_token)
            if judge.sum().item() == judge.numel():
                token_ids = token_ids[:i,:]
                break
            new_id[judge] = 0
            token_ids[i,:] = new_id
        pred = token_ids[1:,:]
        row = []
        for s,t,v in zip(source.T,target.T,pred.T):
            x = [self.id2sm[j.item()] for j in s]
            y = [self.id2sm[j.item()] for j in t]
            p = [self.id2sm[j.item()] for j in v]
            x_str = "".join(x[1:]).split(self.id2sm[2])[0].replace("R","Br").replace("L","Cl")
            y_str = "".join(y[1:]).split(self.id2sm[2])[0].replace("R","Br").replace("L","Cl")
            p_str = "".join(p).split(self.id2sm[2])[0].replace("R","Br").replace("L","Cl")
            judge = True if y_str == p_str else False
            row.append([x_str,y_str,p_str,judge])
        return row
    
    def evaluate(self,args,test_data):
        self.model.eval()
        res = []
        with torch.no_grad():
            for source, target in test_data:
                res.extend(self._eval_batch(source,target,args.device))
        pred_df = pd.DataFrame(res,columns=["input","answer","predict","judge"])
        accuracy = len(pred_df.query("judge == True")) / len(pred_df)
        return pred_df, accuracy
    
def main():
    args = get_args()
    test_loader = prep_valid_data(args)
    model = TransformerLatent(args)
    evaluator = Evaluator(model,args)
    results, accuracy = evaluator.evaluate(args,test_loader)
    results.to_csv(os.path.join(args.experiment_dir,"evaluate_result.csv"))
    print("perfect accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()