import os

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.training.trainer import edge_score
from utils.config import RESULTS_DIR, SEED, DATASET_PATH
import pandas as pd


def plot_edge_prediction_graph(G, model, x, edge_index, edge_weight, node_id_to_idx, model_name, save=False):
    with torch.no_grad():
        embeddings, leader_score = model(x, edge_index, edge_weight)

    edge_probs = {(u, v): torch.sigmoid(edge_score(node_id_to_idx[u], node_id_to_idx[v], embeddings)).item() for u, v in G.edges}

    edges, probs = zip(*edge_probs.items())
    vmin, vmax = min(probs), max(probs)

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=SEED)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=probs, edge_cmap=plt.cm.viridis, edge_vmin=vmin, edge_vmax=vmax, width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Link probability")

    plt.title("Link prediction")
    plt.axis("off")

    if save:
        plt.savefig(os.path.join(RESULTS_DIR, f'{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_{model_name}.png'), bbox_inches='tight', dpi=300)

    plt.show()
    plt.close(fig)


def draw(G, measures, measure_name, save=False):
    values = np.array(list(measures.values()))
    pos = nx.spring_layout(G, seed=SEED)
    vmin, vmax = values.min(), values.max()

    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_nodes(
        G, pos, node_size=300, node_color=values, nodelist=list(measures.keys()),
        cmap=plt.cm.plasma, ax=ax
    )

    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)

    norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=measure_name)

    if save:
        fig.savefig(os.path.join(RESULTS_DIR, f"{measure_name}_plot.png"), dpi=300, bbox_inches='tight')

    plt.title(measure_name)
    plt.axis('off')
    plt.show()


def draw_communities(G, cluster_info, save=False):
    fig = plt.figure(figsize=(12, 10))

    num_clusters = len(cluster_info)
    cmap = plt.colormaps["tab20"].resampled(num_clusters)

    pos = nx.spring_layout(G, seed=SEED)

    for i, c in enumerate(cluster_info):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=c["followers"],
            node_size=300,
            node_color=[cmap(i)] * len(c["followers"]),
            alpha=0.85
        )

    leaders = [c["leader"] for c in cluster_info]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=leaders,
        node_size=800,
        node_color="gold",
        edgecolors="red",
        linewidths=2.5
    )

    if save:
        fig.savefig(os.path.join(RESULTS_DIR, "clusters_plot.png"), dpi=300, bbox_inches='tight')

    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Cluster")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_models_comparison(df_gla, df_gcn, dataset_path, results_dir, gla_name="GraphLA", gcn_name="GraphGCN"):
    df_gla = df_gla.rename(columns={0: gla_name})
    df_gcn = df_gcn.rename(columns={0: gcn_name})

    df_combined = pd.concat([df_gla, df_gcn], axis=1)

    improvement = ((df_combined[gla_name] - df_combined[gcn_name]) /
                   df_combined[gcn_name] * 100).round(2).astype(str) + "%"

    df_final = df_combined.assign(**{"Improvement (%)": improvement})

    print("\nFinal comparison summary\n", df_final)

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]

    csv_path = os.path.join(results_dir, f"{base_name}_models_comparison.csv")
    df_final.to_csv(csv_path, index=True)

    metrics = df_combined.index
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df_combined[gla_name], width=width, label=gla_name)
    plt.bar(x + width/2, df_combined[gcn_name], width=width, label=gcn_name)

    plt.xticks(x, metrics, rotation=45)
    plt.ylabel('Score')
    plt.title(f'Comparison of {gla_name} vs {gcn_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()

    graph_path = os.path.join(results_dir, f"{base_name}_models_comparison_graph.png")
    plt.savefig(graph_path, dpi=300)
    plt.show()

    return df_final
