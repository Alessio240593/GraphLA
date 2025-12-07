import pickle

import networkx as nx

from utils.config import DATASET_PATH, CLUSTERS_PATH, SEED
from utils.plot import draw_communities
from utils.utility import load_graph

# --- 1) Load graph ---
G = load_graph(DATASET_PATH)

# --- 2) Clustering
cluster = nx.algorithms.community.louvain.louvain_communities(
    G, weight='weight', resolution=0.45, threshold=1e-07, seed=SEED
)

cluster_info = []
for i, comm in enumerate(cluster, start=1):
    subG = G.subgraph(comm)
    closeness = nx.closeness_centrality(subG, distance='weight')
    leader = max(closeness, key=closeness.get)
    followers = [n for n in comm if n != leader]

    cluster_info.append({
        "id": i,
        "leader": leader,
        "followers": followers,
    })

# --- 3) Show plot
draw_communities(G, cluster_info, save=True)

# --- 4)Save the cluster data
print("Saving clusters information...")

try:
    with open(CLUSTERS_PATH, "wb") as f:
        pickle.dump(cluster_info, f)

except (OSError, IOError) as e:
    print(f"Some error occur: {e}")
else:
    print("Clusters information saved successfully")