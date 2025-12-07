import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.config import SEED


def split_edges_train_val_test(edges_all, labels, test_size=0.2, random_state=SEED):
    edges_list = list(edges_all)
    labels_list = list(labels)

    train_val_edges, test_edges, train_val_labels, test_labels = train_test_split(
        edges_list, labels_list, test_size=test_size, random_state=random_state
    )

    train_val_edges = torch.tensor([list(e) for e in train_val_edges], dtype=torch.long)
    test_edges = torch.tensor([list(e) for e in test_edges], dtype=torch.long)

    train_val_labels = torch.tensor(train_val_labels, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    return train_val_edges, train_val_labels, test_edges, test_labels


def edge_score(u, v, embeddings):
    u = int(u)
    v = int(v)
    return (embeddings[u] * embeddings[v]).sum()


def compute_loss(model, x, edge_index, edge_weight, edges, labels):
    embeddings, leader_score = model(x, edge_index, edge_weight)
    preds = torch.stack([edge_score(u, v, embeddings) for u, v in edges])

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(preds, labels)

    preds_sigmoid = torch.sigmoid(preds)
    return loss, preds_sigmoid


def evaluate_on_edges(model, x, edge_index, edge_weight, edges, labels):
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(x, edge_index, edge_weight)
        preds = torch.stack([torch.sigmoid((embeddings[int(u)] * embeddings[int(v)]).sum()) for u, v in edges])

        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(preds, labels)

        from sklearn.metrics import precision_recall_curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(labels.numpy(), preds.numpy())
        f1_scores = 2 * precision_curve[:-1] * recall_curve[:-1] / (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
        best_threshold = thresholds[f1_scores.argmax()]

        pred_labels = (preds >= best_threshold).float()
        labels_np = labels.numpy()
        pred_labels_np = pred_labels.numpy()

        auc = roc_auc_score(labels_np, preds.numpy())
        ap = average_precision_score(labels_np, preds.numpy())
        precision = precision_score(labels_np, pred_labels_np)
        recall = recall_score(labels_np, pred_labels_np)

    return {
        "AUC": auc,
        "AP": ap,
        "Precision": precision,
        "Recall": recall,
        "Loss": loss.item()
    }


def train_kfold(
        model_class, base_params, x, edge_index, edge_weight,
        edges_train_val, labels_train_val,
        edges_test, labels_test,
        param_grid, k=5, epochs=20
):
    best_ap = -1
    best_params = None
    results = []

    for params in param_grid:
        lr = params.get("lr", 0.01)
        h_feats = params.get("hidden_feats", base_params.get("hidden_feats", 32))

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        aps = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(edges_train_val)):
            model_fold = model_class(in_feats=base_params['in_feats'], hidden_feats=h_feats)

            optimizer = torch.optim.Adam(model_fold.parameters(), lr=lr, weight_decay=5e-4)

            edges_train, labels_train = edges_train_val[train_idx], labels_train_val[train_idx]
            edges_val, labels_val = edges_train_val[val_idx], labels_train_val[val_idx]

            for epoch in range(epochs):
                model_fold.train()
                optimizer.zero_grad()
                train_loss, _ = compute_loss(model_fold, x, edge_index, edge_weight, edges_train, labels_train)
                train_loss.backward()
                optimizer.step()

            metrics = evaluate_on_edges(model_fold, x, edge_index, edge_weight, edges_val, labels_val)
            aps.append(metrics['AP'])

        avg_ap = sum(aps) / k
        results.append({**params, "avg_ap": avg_ap})

        if avg_ap > best_ap:
            best_ap = avg_ap
            best_params = params

    final_model = model_class(in_feats=base_params['in_feats'], hidden_feats=best_params['hidden_feats'])
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

    for epoch in tqdm(range(epochs), desc="Training model"):
        final_model.train()
        optimizer.zero_grad()
        train_loss, _ = compute_loss(final_model, x, edge_index, edge_weight, edges_train_val, labels_train_val)
        train_loss.backward()
        optimizer.step()

    test_metrics = evaluate_on_edges(final_model, x, edge_index, edge_weight, edges_test, labels_test)

    return final_model, best_params, test_metrics
