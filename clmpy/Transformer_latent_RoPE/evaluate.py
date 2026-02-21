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
from ..get_args import get_argument


class Evaluator():
    def __init__(self,args,model):
        self.args = args
        self.id2sm = args.token.id2sm
        self.model = model.to(args.device)
        self.maxlen = args.n_positions
        self.device = args.device
        if len(args.model_path) > 0:
            self._load(args.model_path)

    def _load(self,path):
        self.model.load_state_dict(torch.load(path))
    
    def _eval_batch(self, source, target):
        source = source.to(self.device)
        latent = self.model.encoder(source)
        batch_size = source.size(1)

        token_ids = torch.zeros((self.maxlen, batch_size), dtype=torch.long).to(self.device)
        token_ids[0, :] = 1 # Start Token
        
        # KVキャッシュの初期化
        past = None
        
        # 現在の入力トークン (最初のステップは Start Token のみ)
        current_input = token_ids[0:1, :] # Shape: [1, B]

        for i in range(1, self.maxlen):
            # 【高速化】
            # 全系列ではなく、直前のトークン(current_input)と過去の記憶(past)を渡す
            out, past = self.model.decoder(current_input, latent, layer_past=past)
            
            # out は [1, B, Vocab] (入力長が1なので)
            _, out_id = out.max(dim=2) 
            new_id = out_id[0, :] # Shape: [B]
            
            # 終了判定
            is_end_token = token_ids[i-1, :] == 2
            is_pad_token = token_ids[i-1, :] == 0
            judge = torch.logical_or(is_end_token, is_pad_token)
            
            if judge.sum().item() == judge.numel():
                # 全バッチ終了
                token_ids = token_ids[:i, :] # ここまでの結果を保持
                break
            
            new_id[judge] = 0
            token_ids[i, :] = new_id
            
            # 次のステップへの入力を用意
            current_input = new_id.unsqueeze(0) # Shape: [1, B]

        pred = token_ids[1:, :]
        
        # 結果の整形処理（変更なし）
        row = []
        for s, t, v in zip(source.T, target.T, pred.T):
            # 辞書のキーエラー回避のため .item() を明示
            x = [self.id2sm[j.item()] for j in s]
            y = [self.id2sm[j.item()] for j in t]
            p = [self.id2sm[j.item()] for j in v]
            
            # 特殊トークン以降を削除
            x_str = "".join(x[1:]).split(self.id2sm[2])[0].replace("R", "Br").replace("L", "Cl")
            y_str = "".join(y[1:]).split(self.id2sm[2])[0].replace("R", "Br").replace("L", "Cl")
            p_str = "".join(p).split(self.id2sm[2])[0].replace("R", "Br").replace("L", "Cl")
            
            judge = True if y_str == p_str else False
            row.append([x_str, y_str, p_str, judge])
            
        return row
    
    def evaluate(self,test_data):
        self.model.eval()
        res = []
        test_data = prep_valid_data(self.args,test_data)
        with torch.no_grad():
            for source, target in test_data:
                res.extend(self._eval_batch(source,target))
        pred_df = pd.DataFrame(res,columns=["input","answer","predict","judge"])
        accuracy = len(pred_df.query("judge == True")) / len(pred_df)
        return pred_df, accuracy
    
def main():
    args = get_argument()
    test_data = pd.read_csv(args.test_path,index_col=0)
    model = TransformerLatent(args)
    evaluator = Evaluator(args,model)
    results, accuracy = evaluator.evaluate(test_data)
    results.to_csv(os.path.join(args.experiment_dir,"evaluate_result.csv"))
    print("perfect accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()