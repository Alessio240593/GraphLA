import pickle
import random
import sys
from os import path

import networkx as nx
import torch

from utils.config import CLUSTERS_PATH, NEG_EDGES_PATH, POS_EDGES_PATH


def load_model(model, model_path):
    if not path.isfile(model_path):
        print(f"Error: the model '{model_path}' doesn't exists, please generate it with model_training.py file")
        sys.exit(1)

    if path.getsize(model_path) == 0:
        print(f"Error: the model '{model_path}' is empty")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        sys.exit(1)


def prepare_graph_inputs(graph_file):
    G = load_graph(graph_file)
    node_list = list(G.nodes)
    node_id_to_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.tensor([
        [G.degree(n),
         sum(G[n][nbr]["weight"] for nbr in G.neighbors(n))]
        for n in G.nodes
    ], dtype=torch.float)

    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

    edge_index = torch.tensor([[node_id_to_idx[u], node_id_to_idx[v]] for u, v in G.edges],
                              dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([G[u][v]["weight"] for u, v in G.edges], dtype=torch.float)

    return G, x, edge_index, edge_weight


def prepare_training_set(pos_edges, possible_neg_edges):
    num_neg = len(pos_edges)
    neg_edges = random.sample(possible_neg_edges, min(num_neg, len(possible_neg_edges)))

    edges_all = pos_edges + neg_edges
    labels_all = [1] * len(pos_edges) + [0] * len(neg_edges)

    combined = list(zip(edges_all, labels_all))
    random.shuffle(combined)
    edges_all, labels_all = zip(*combined)

    edges_all = torch.tensor(edges_all, dtype=torch.long)
    labels_all = torch.tensor(labels_all, dtype=torch.float)

    return edges_all, labels_all


def load_graph(file_path: str) -> nx.Graph:
    if not path.exists(file_path):
        raise FileNotFoundError(f"File not found {file_path}")

    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            w = float(parts[2])
            G.add_edge(u, v, weight=w)

    return G


def load_cluster_info(clusters_path: str = CLUSTERS_PATH):
    with open(clusters_path, "rb") as f:
        return pickle.load(f)


def load_training_set(pos_path: str = POS_EDGES_PATH, neg_path: str = NEG_EDGES_PATH):
    with open(pos_path, "rb") as f:
        pos_edges = pickle.load(f)

    with open(neg_path, "rb") as f:
        possible_neg_edges = pickle.load(f)

    return pos_edges, possible_neg_edges
