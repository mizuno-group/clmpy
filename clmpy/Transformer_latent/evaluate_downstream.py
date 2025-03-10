# -*- coding: utf-8 -*-
# 240620

import os
from argparse import ArgumentParser, FileType
import yaml

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from .model import TransformerLatent_MLP
from ..preprocess import *
import matplotlib.pyplot as plt
import seaborn as sns


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
    args.experiment_dir = "/".join(args.model_path.split("/")[:-1])
    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

class Evaluator_MLP():
    def __init__(self,model,args):
        self.args = args
        self.id2sm = args.token.id2sm
        self.model = model.to(args.device)
        self.task = args.task
        if len(args.model_path) > 0:
            self._load(args.model_path)

    def _load(self,path):
        self.model.load_state_dict(torch.load(path), strict=False)



    def _eval_batch(self, source, y, device):
        source = source.to(device)
        latent = self.model.encoder(source)
        out_d = self.model.mlp(latent)
        
        y = y.to(device)
        y = y.unsqueeze(1) 
        d_rounded = torch.round(torch.sigmoid(out_d)).long()
        judge = torch.eq(y, d_rounded).squeeze(1)  
        row = []
        for s, t, r, d, j in zip(source.T, y, d_rounded, out_d, judge):
            x = [self.id2sm[j.item()] for j in s]  # SMILES 変換
            x_str = "".join(x[1:]).split(self.id2sm[2])[0].replace("R", "Br").replace("L", "Cl")
            
            row.append([x_str, d.item(),r.item(), t.item(), j.item()])  

        return row

    def evaluate(self,test_data):
        self.model.eval()
        res = []
        test_data = prep_valid_data(self.args,test_data,downstream=True)
        with torch.no_grad():
            for source, target, y in test_data:
                res.extend(self._eval_batch(source,y,self.args.device))
        pred_df = pd.DataFrame(res,columns=["input","predict","round","answer","judge"])
        
        if self.task == "regression":
            mae = mean_absolute_error(pred_df["answer"], pred_df["predict"])
            mse = mean_squared_error(pred_df["answer"], pred_df["predict"])
            rmse = np.sqrt(mse)
            r2 = r2_score(pred_df["answer"], pred_df["predict"])

            print(f"MAE (Mean Absolute Error): {mae:.4f}")
            print(f"MSE (Mean Squared Error): {mse:.4f}")
            print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
            print(f"R² Score: {r2:.4f}")

            # 散布図を描画
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=pred_df["answer"], y=pred_df["predict"], alpha=0.6)
            plt.plot([pred_df["answer"].min(), pred_df["answer"].max()], 
                    [pred_df["answer"].min(), pred_df["answer"].max()], 
                    color="red", linestyle="--")  # y=xの基準線
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs Predicted")
            plt.grid(True)

            # 画像を保存
            plt.savefig(os.path.join(self.args.experiment_dir, "regression_results.png"), dpi=300, bbox_inches="tight")
            plt.close()
            return pred_df

        else:
            pred_df["predict"] = 1 / (1 + np.exp(-pred_df["predict"]))
            TP = len(pred_df.query("round == True and answer == True"))
            TN = len(pred_df.query("round == False and answer == False"))
            FP = len(pred_df.query("round == True and answer == False"))
            FN = len(pred_df.query("round == False and answer == True"))
            
            auroc = roc_auc_score(pred_df["answer"], pred_df["predict"])
            accuracy = (TP + TN) / len(pred_df) if len(pred_df) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
            false_negative_rate = FN / (TP + FN) if (TP + FN) > 0 else 0

            conf_matrix_data = [[TP, FN], [FP, TN]]
            print(f"AUROC: {auroc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall (Sensitivity): {recall:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")
            print(f"False Negative Rate (FNR): {false_negative_rate:.4f}")
        
            labels = ["Positive", "Negative"]
            plt.figure(figsize=(5, 4))
            sns.heatmap(conf_matrix_data, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Predicted Positive", "Predicted Negative"],
                        yticklabels=["Actual Positive", "Actual Negative"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(self.args.experiment_dir,"confusion_matrix.png"), dpi=300, bbox_inches="tight")
            plt.close()

            return pred_df
    
def main():
    args = get_args()
    test_data = pd.read_csv(args.test_path,index_col=0)
    test_data = test_data[test_data["Class"] == "test"]
    model = TransformerLatent_MLP(args)
    evaluator = Evaluator_MLP(model,args)
    results = evaluator.evaluate(test_data)
    results.to_csv(os.path.join(args.experiment_dir,"evaluate_result_mlp.csv"))
    

if __name__ == "__main__":
    main()