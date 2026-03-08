#!/bin/sh
#PBS -q short-g
#PBS -l select=1
#PBS -W group_list=XXXX

# モジュールロード
module load cuda/12.4
module load cudnn/9.5.1.17
module load cmake/3.31.1
module load pytorch-gpu/2.5.1

# パスの設定
NEW_ENV_PATH="/work/XX/XXX/env/pyenv/418"


rm -rf "$NEW_ENV_PATH"
python3 -m venv --system-site-packages "$NEW_ENV_PATH"

source "$NEW_ENV_PATH/bin/activate"

export PYTHONPATH="$NEW_ENV_PATH/lib64/python3.9/site-packages:$NEW_ENV_PATH/lib/python3.9/site-packages:$PYTHONPATH"

python3 -m pip install --upgrade pip


python3 -m pip install --no-deps /work/XX/XXX/clmpy

python3 -m pip install pandas pyyaml wandb transformers==4.18.0
python3 -m pip install numpy pandas matplotlib scipy seaborn rdkit PyYAML wandb
python3 -m pip install torch torchvision torchaudio
python3 -m pip install tqdm optuna plotly scikit-learn gensim pubchempy xgboost lightgbm
python3 -m pip install --upgrade typing_extensions pydantic
python3 -m pip install transformers==4.18.0

python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# --- 4. 実行 ---
export WANDB_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#clmpy.transformerlatent_mix.train --config   /work/gd43/a97007/clmpy-main/result/251225_mix26/TransformerVAE.ym
#clmpy.transformerlatent_mix.train --config   /work/gd43/a97007/clmpy-main/result/251225_mix44/TransformerVAE.yml
clmpy.transformerlatent_mix.train --config   /work/gd43/a97007/clmpy-main/result/260305_mix62/TransformerVAE.yml
#clmpy.transformerlatent_mix.train --config   /work/gd43/a97007/clmpy-main/result/251225_mix08/TransformerVAE.yml
#clmpy.transformerlatent.evaluate --config /work/gd43/a97007/clmpy-main/result/260220_Standard_418/TransformerVAE.yml --model_path /work/gd43/a97007/clmpy-main/result/260220_Standard_418/best_model.pt --test_path /work/gd43/a97007/clmpy-main/zinc_test.csv
