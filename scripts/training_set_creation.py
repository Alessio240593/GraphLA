import pickle

from utils.config import DATASET_PATH, POS_EDGES_PATH, NEG_EDGES_PATH
from utils.utility import load_graph, load_cluster_info

# --- 1) Load the graph
G = load_graph(DATASET_PATH)

# --- 2) Load cluster information
cluster_info = load_cluster_info()

# --- 3) Create the training set
node_list = list(G.nodes)
node_id_to_idx = {n: i for i, n in enumerate(node_list)}

followers = [f for c in cluster_info for f in c["followers"]]
leaders = [c["leader"] for c in cluster_info]

pos_edges = [
    (node_id_to_idx[f], node_id_to_idx[l])
    for l in leaders
    for f in followers
    if G.has_edge(f, l)
]

all_pairs = [(node_id_to_idx[f], node_id_to_idx[l])
             for l in leaders
             for f in followers
             if f != l]

possible_neg_edges = list(set(all_pairs) - set(pos_edges))

# --- 4) Save the training set
print("Saving training set...")

try:
    with open(POS_EDGES_PATH, "wb") as f:
        pickle.dump(pos_edges, f)

    with open(NEG_EDGES_PATH, "wb") as f:
        pickle.dump(possible_neg_edges, f)

except (OSError, IOError) as e:
    print(f"Some error occur: {e}")
else:
    print("Training set saved successfully")
