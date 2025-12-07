# GraphLA Link Prediction

This repository contains the implementation of **GraphLA**, a leader-aware GNN for link prediction on social networks.

## Running in Google Colab
1. Copy the folder `notebook/GraphLA` to your Google Drive.  
2. Open the `main.ipynb` notebook in Colab.
3. Run all cells (the notebook will execute all scripts automatically).

## Running Locally
```bash
git clone <repo_url>
cd GraphLA

# Using venv
python3 -m venv GraphLA
source GraphLA/bin/activate 

pip install --upgrade pip
pip install -r requirements.txt

python main.py

deactivate

# Using conda
conda create -n GraphLA python=3.x
conda activate GraphLA

python main.py

conda deactivate
```

