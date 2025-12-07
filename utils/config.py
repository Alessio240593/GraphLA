import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DIRS = {
    "dataset": "dataset",
    "data": "data",
    "models": "saved_models",
    "results": "results"
}


def get_path(name: str) -> str:
    path = os.path.join(PROJECT_ROOT, _DIRS[name])
    os.makedirs(path, exist_ok=True)
    return path


DATASET_DIR = get_path("dataset")
DATA_DIR = get_path("data")
MODELS_DIR = get_path("models")
RESULTS_DIR = get_path("results")

DATASET_PATH = os.path.join(DATASET_DIR, "mammalia-dolphin-florida-overall.edges")
CLUSTERS_PATH = os.path.join(DATA_DIR, f'{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_clusters_info.pkl')
POS_EDGES_PATH = os.path.join(DATA_DIR, f'{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_positive_edges.pkl')
NEG_EDGES_PATH = os.path.join(DATA_DIR, f'{os.path.splitext(os.path.basename(DATASET_PATH))[0]}_negative_edges.pkl')

SEED = 42