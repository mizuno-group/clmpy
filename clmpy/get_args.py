import os
from argparse import ArgumentParser, FileType
import yaml

from .preprocess import prep_token

def get_argument(notebook=False,config=""):
    parser = ArgumentParser()

    # general
    parser.add_argument("--config",type=str,default="")
    parser.add_argument("--seed",type=int,default=123)
    parser.add_argument("--num_workers",type=int,default=1)
    parser.add_argument("--experiment_dir",type=str,default=os.getcwd())
    parser.add_argument("--device",type=str,choices=["cpu","cuda"],default="cpu")

    # data processing
    parser.add_argument("--train_path",type=str,default=None)
    parser.add_argument("--valid_path",type=str,default=None)
    parser.add_argument("--token_path",type=str,default="../sample/tokens/normal_tokens.txt")
    parser.add_argument("--bucketing",type=bool,default=True)
    parser.add_argument("--bucket_min",type=int,default=10)
    parser.add_argument("--bucket_max",type=int,default=150)
    parser.add_argument("--bucket_step",type=int,default=10)
    parser.add_argument("--batch_shuffle",type=bool,default=True)
    parser.add_argument("--SFL",type=bool,default=False)

    # GRU model architecture
    parser.add_argument("--embedding_dim",type=int,default=128)
    parser.add_argument("--enc_gru_layer",type=list,default=[256,512,1024])
    parser.add_argument("--latent_dim",type=int,default=256)
    parser.add_argument("--dec_gru_layer",type=list,default=[256,512,1024])

    # transformer model architecture
    parser.add_argument("--transformer_dim",type=int,default=256)
    parser.add_argument("--n_layer",type=int,default=8)
    parser.add_argument("--n_head",type=int,default=8)
    parser.add_argument("--n_positions",type=int,default=500)
    parser.add_argument("--layer_norm_epsilon",type=float,default=1.0e-5)

    # train
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--steps",type=int,default=500000)
    parser.add_argument("--dropout",type=float,default=0.1)
    parser.add_argument("--max_lr",type=float,default=1.0e-1)
    parser.add_argument("--warmup_step",type=int,default=8000)
    parser.add_argument("--beta",type=float,default=1.0)
    parser.add_argument("--valid_step_range",type=int,default=500)
    parser.add_argument("--patience_step",type=int,default=10000)

    # train downstream
    parser.add_argument("--mlp_dim",type=list,default=[128])
    parser.add_argument("--gamma",type=float,default=1.0)
    parser.add_argument("--task",type=str,choices=["classification","regression"])

    # evaluation
    parser.add_argument("--test_path",type=str,default="")
    parser.add_argument("--model_path",type=str,default="")
    parser.add_argument("--maxlen",type=int,default=500)

    # result handling
    parser.add_argument("--loss_log",type=bool,default=True)
    #parser.add_argument("--loss_plot",type=bool,default=True)

    # generate / encode
    parser.add_argument("--latent_path",type=str,default="")
    parser.add_argument("--smiles_path",type=str,default="")
    parser.add_argument("--output_path",type=str,default="")
    
    if notebook == True:
        args = parser.parse_args(["--config",config])
    else:
        args = parser.parse_args()
    if args.config != "":
        with open(args.config,"r") as c:
            config_dict = yaml.load(c,Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            arg_dict[key] = value

    args.token = prep_token(args.token_path)
    args.vocab_size = args.token.length
    args.patience = args.patience_step // args.valid_step_range
    return args