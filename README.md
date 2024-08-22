# clmpy

A platform of Chemical Language Model (CLM) for comparing translation, generation and description.

# Note

This repository is under construction and will be officially released by Mizuno group.
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.

## How to use

### Training
``` shell
python3 clmpy.gru.train --config config.yml
```

### evaluation
``` shell
python3 clmpy.gru.evaluate \
    --config config.yml \
    --model_path <trained model path> \
    --test_path <test data path> 
```

### generation
``` shell
python3 clmpy.gru.generate \
    --config config.yml \
    --model_path <trained model path> \
    --latent_path <latent descriptor path (.csv)> 
```

### encodeing
``` shell
python3 clmpy.gru.encode \
    --config config.yml \
    --model_path <trained model path> \
    --smiles_path <smiles_list_path (Line separated txt file)> 
```

## run interactively (.ipynb)
``` python
!python3 -m pip install clmpy
from clmpy.GRU.model import GRU
from clmpy.GRU.train import Trainer
from clmpy.preprocess import *

args = get_notebook_args(<path to config.yml>)
train_loader = prep_train_data(args)
valid_loader = prep_valid_data(args)
model = GRU(args)
criteria, optimizer, scheduler, es = load_train_objs_gru(args,model)
# possible with self-defined objects
trainer = Trainer(args,model,train_loader,valid_loader,criteria,optimizer,scheduler,es)
loss_t, loss_v = trainer.train(args)
torch.save(trainer.best_model.state_dict(),<model path>))
```