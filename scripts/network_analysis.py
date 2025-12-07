import os
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.config import DATASET_PATH, RESULTS_DIR
from utils.plot import draw
from utils.utility import load_graph


# --- 1) Load graph ---
G = load_graph(DATASET_PATH)

# --- 2) Define measures ---
measures = {
    "Closeness centrality": nx.closeness_centrality(G, distance='weight'),
    "Betweenness centrality": nx.betweenness_centrality(G, weight='weight'),
    "Local clustering value": nx.clustering(G, weight='weight')
}

# --- 3) Compute node-level stats ---
summary_rows = []
for name, values in measures.items():
    vals = np.array(list(values.values()))

    draw(G, values, name, save=True)

    summary_rows.append({
        "measure": name,
        "min": vals.min(),
        "max": vals.max(),
        "mean": vals.mean(),
        "std": vals.std()
    })

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(name)

    axes[0].boxplot(vals, vert=False)
    axes[0].set_xlabel("Value")
    axes[0].set_yticks([])

    axes[1].hist(vals, bins=20, color='skyblue', edgecolor='black')
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Number of nodes")

    fig.savefig(os.path.join(RESULTS_DIR, f"{name}_boxplot_hist.png"), dpi=300, bbox_inches='tight')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- 4) Add global graph-level measures ---
summary_rows.append({
    "measure": "Average clustering coefficient",
    "min": '',
    "max": '',
    "mean": nx.average_clustering(G, weight='weight'),
    "std": ''
})
summary_rows.append({
    "measure": "Global clustering coefficient",
    "min": '',
    "max": '',
    "mean": nx.transitivity(G),
    "std": ''
})

# --- 5) Create DataFrame ---
df_summary = pd.DataFrame(summary_rows)

print(df_summary)

print("\nSaving the analysis of the graph...")

# --- 6) Save to CSV ---
df_summary.to_csv(path.join(RESULTS_DIR, f'{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_graph_analysis.csv'), index=False)

print("The analysis of the graph is saved successfully...")