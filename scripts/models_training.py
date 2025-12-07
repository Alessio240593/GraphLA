import os
import random
from os import path

import numpy as np
import pandas as pd
import torch

from src.models.Graph_LA import GraphLA
from src.models.Standard_GCN import StandardGCN
from src.training.trainer import train_kfold, split_edges_train_val_test
from utils.config import DATASET_PATH, MODELS_DIR, RESULTS_DIR, SEED
from utils.plot import plot_edge_prediction_graph, plot_models_comparison
from utils.utility import load_training_set, prepare_training_set, prepare_graph_inputs

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 1) Create input data for GNN
G, x, edge_index, edge_weight = prepare_graph_inputs(DATASET_PATH)
node_list = list(G.nodes)
node_id_to_idx = {n: i for i, n in enumerate(node_list)}

# --- 2) Load and prepare training set
pos_edges, possible_neg_edges = load_training_set()
edges_all, labels_all = prepare_training_set(pos_edges, possible_neg_edges)
train_val_edges, train_val_labels, test_edges, test_labels = split_edges_train_val_test(edges_all, labels_all)
base_params = {"in_feats": x.size(1)}
param_grid = [
    {"lr": 0.01, "hidden_feats": 16},
    {"lr": 0.01, "hidden_feats": 32},
    {"lr": 0.005, "hidden_feats": 16},
    {"lr": 0.005, "hidden_feats": 32}
]

# --- 3) Train, evaluate and test GraphLA

# GraphLA
print("GraphLA model training in progress...")
final_model, _, test_metrics = train_kfold(
    GraphLA,
    base_params,
    x, edge_index, edge_weight,
    train_val_edges, train_val_labels,
    test_edges, test_labels,
    param_grid,
    k=5,
    epochs=50
)

df_test_GLA = pd.DataFrame([test_metrics]).T
print(f'\nTest summary \n {df_test_GLA}')

plot_edge_prediction_graph(G, final_model, x, edge_index, edge_weight, node_id_to_idx, "graphLA", save=True)

os.makedirs(MODELS_DIR, exist_ok=True)
print("Saving GraphLA model...")
torch.save(final_model.state_dict(), path.join(MODELS_DIR, f'{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_graphLA.pt'))
print("Model GraphLA saved successfully...")

# --- 4) Train, evaluate and test StandardGCN

# Standard GCN
print("\nGCN model training in progress...")
final_model, _, test_metrics = train_kfold(
    StandardGCN,
    base_params,
    x, edge_index, edge_weight,
    train_val_edges, train_val_labels,
    test_edges, test_labels,
    param_grid,
    k=5,
    epochs=50
)

df_test_GCN = pd.DataFrame([test_metrics]).T
print(f'\nTest summary\n {df_test_GCN}')

plot_edge_prediction_graph(G, final_model, x, edge_index, edge_weight, node_id_to_idx, "GCN", save=True)

os.makedirs(MODELS_DIR, exist_ok=True)
print("Saving GraphGCN model...")
torch.save(final_model.state_dict(), path.join(MODELS_DIR, f'{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_graphGCN.pt'))
print("Model GraphGCN saved successfully...")

# --- 5) Final comparison
df_final = plot_models_comparison(
    df_test_GLA,
    df_test_GCN,
    DATASET_PATH,
    RESULTS_DIR
)