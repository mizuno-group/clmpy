# clmpy

A platform of Chemical Language Model (CLM) for comparing translation, generation and description.

# Note

This repository is under construction and will be officially released by Mizuno group.
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.

## How to use

### linux terminal
``` shell
python3 clmpy.gru.train --config config.yml
python3 clmpy.gru.evaluate \
    --config config.yml \
    --model_path <path to model.pt> \
    --test_path <path to test_data.csv>
python3 clmpy.gru.generate \
    --config config.yml \
    --model_path <path to model.pt> \
    --latent_path <path to latent matrix.csv>
python3 clmpy.gru.encode \
    --config config.yml \
    --model_path <path to model.pt> \
    --smiles_path <path to smiles.txt>
```

## run interactively (.ipynb)
``` python
!python3 -m pip install clmpy
from clmpy.GRU.model import GRU
from clmpy.GRU.train import Trainer as GRUTrain
from clmpy.GRU.evaluate import Evaluator as GRUEval
from clmpy.GRU.generate import Generator as GRUGen
from clmpy.GRU.encode import encode as gruenc
from clmpy.preprocess import *
from clmpy.utils import *
from clmpy.get_args import get_argument

args = get_argument(notebook=True,config=<path to config.yml>)

# training
train_data = pd.read_csv(<path to train_data.csv>,index_col=0)
valid_data = pd.read_csv(<path to valid_data.csv>,index_col=0)
# Column names should be "input" and "output"
model = GRU(args)
criteria, optimizer, scheduler, es = load_train_objs(args,model)
# possible with self-defined objects
trainer = GRUTrain(args,model,train_data,valid_data,criteria,optimizer,scheduler,es)
trainer.train()
torch.save(trainer.best_model.state_dict(),<model path>)

# evaluation
test_data = pd.read_csv(<path to test_data.csv>,index_col=0)
evaluator = GRUEval(args,trainer.best_model)
res, acc = evaluator.evaluate(test_data)

# encoding
model = GRU(args).to(args.device)
model.load_state_dict(torch.load(<model.pt>))
latent = gruenc(args,<list of smiles>,model)

# structure generaton
latent = pd.read_csv(<path to latent.csv>)
generator = GRUGen(args,model)
generated = generator.generate(latent)
```