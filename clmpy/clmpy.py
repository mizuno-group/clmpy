# -*- coding: utf-8 -*-
# 240520

import yaml
import torch

from utils import attrdict

class CLMpy:
    def __init__(self,config=None,config_path=None):
        if config is None:
            if config_path is not None:
                with open(config_path,"r") as f:
                    config = yaml.safe_load(f)
            else:
                config = attrdict()
        default_config = {
            "batch_size": 512,
            "num_workers": 2,
            "token_path": "data/SFL_tokens.txt",
            "vocab_size": 665,
            "embedding_dim": 128,
            "enc_gru_layer": [256,512,1024],
            "latent_dim": 256,
            "dec_gru_layer": [256,512,1024],
            "epochs": 100,
            "plot": True,
            "dropout": 0.1,
            "lr": 1e-5,
            "gamma": 0.9,
            "patience": 10,
            "beta": 0,
            "buckets_min": 20,
            "buckets_max": 200,
            "buckets_step": 10,
            "train_data": "data/train.csv",
            "valid_data": "data/val_100k.csv",
            "maxlen": 1000
        }
        self.config = {**default_config,**config}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
