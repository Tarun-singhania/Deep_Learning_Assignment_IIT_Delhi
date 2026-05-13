import argparse
import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from model import GCNModel

GraphDict = Dict[str, Any]

# Metrics and loss functions

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    positive_mask = x >= 0
    negative_mask = ~positive_mask
    out = np.empty_like(x, dtype=np.float32)
    out[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[negative_mask])
    out[negative_mask] = exp_x / (1.0 + exp_x)
    return out


def binary_cross_entropy_with_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    pos_weight: Optional[np.ndarray] = None,
) -> float:
    logits = logits.astype(np.float32)
    targets = targets.astype(np.float32)

    if pos_weight is None:
        loss_matrix = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
        return float(np.mean(loss_matrix))

    pos_weight = pos_weight.astype(np.float32)
    log_weight = 1.0 + (pos_weight - 1.0) * targets
    softplus_neg_logits = np.maximum(-logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    loss_matrix = (1.0 - targets) * logits + log_weight * softplus_neg_logits
    return float(np.mean(loss_matrix))


def bce_with_logits_gradient(
    logits: np.ndarray,
    targets: np.ndarray,
    pos_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    logits = logits.astype(np.float32)
    targets = targets.astype(np.float32)
    probs = sigmoid(logits)

    if pos_weight is None:
        grad = probs - targets
    else:
        pos_weight = pos_weight.astype(np.float32)
        grad = probs * (1.0 + (pos_weight - 1.0) * targets) - pos_weight * targets

    grad /= np.prod(targets.shape, dtype=np.float32)
    return grad.astype(np.float32)


def micro_f1_stats_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, int]:
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)

    tp = int(np.sum((preds == 1) & (targets_int == 1)))
    fp = int(np.sum((preds == 1) & (targets_int == 0)))
    fn = int(np.sum((preds == 0) & (targets_int == 1)))
    return {"tp": tp, "fp": fp, "fn": fn}


def micro_f1_from_stats(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2.0 * precision * recall / (precision + recall + 1e-12))

# Dataset loading utilities
def _lookup(mapping: Dict[Any, Any], key: Any) -> Any:
    candidates = [key]
    if isinstance(key, str):
        try:
            candidates.append(int(key))
        except ValueError:
            pass
    else:
        candidates.append(str(key))

    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    raise KeyError(f"Key {key!r} not found in mapping.")


def _label_to_array(label_value: Any) -> np.ndarray:
    if isinstance(label_value, list):
        return np.asarray(label_value, dtype=np.float32)

    if isinstance(label_value, dict):
        ordered_keys = sorted(label_value.keys(), key=lambda x: int(x))
        return np.asarray([label_value[k] for k in ordered_keys], dtype=np.float32)

    if isinstance(label_value, np.ndarray):
        return label_value.astype(np.float32)

    raise TypeError(f"Unsupported label format: {type(label_value)}")


def load_graph_json(graph_path: str) -> Tuple[List[Any], Dict[Any, Dict[str, Any]], Dict[Any, List[Any]]]:
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    if not isinstance(graph_data, dict):
        raise ValueError("ppi-G.json must be a JSON object.")
    if "nodes" not in graph_data:
        raise KeyError("ppi-G.json is missing 'nodes'.")

    edge_key = "links" if "links" in graph_data else "edges" if "edges" in graph_data else None
    if edge_key is None:
        raise KeyError("ppi-G.json must contain either 'links' or 'edges'.")

    node_ids: List[Any] = []
    node_attrs: Dict[Any, Dict[str, Any]] = {}
    adjacency: Dict[Any, List[Any]] = {}

    for node_obj in graph_data["nodes"]:
        node_id = node_obj["id"]
        attrs = {k: v for k, v in node_obj.items() if k != "id"}
        node_ids.append(node_id)
        node_attrs[node_id] = attrs
        adjacency[node_id] = []

    for edge_obj in graph_data[edge_key]:
        src = edge_obj["source"]
        dst = edge_obj["target"]
        adjacency[src].append(dst)
        adjacency[dst].append(src)

    return node_ids, node_attrs, adjacency


def connected_components_manual(node_ids: List[Any], adjacency: Dict[Any, List[Any]]) -> List[List[Any]]:
    visited = set()
    components: List[List[Any]] = []

    for start in node_ids:
        if start in visited:
            continue

        queue = deque([start])
        visited.add(start)
        component: List[Any] = []

        while queue:
            node = queue.popleft()
            component.append(node)

            for nbr in adjacency[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        components.append(component)

    return components


def normalize_adjacency(
    adj: np.ndarray,
    add_self_loops: bool = True,
    normalization: str = "symmetric",
) -> np.ndarray:
    a = adj.astype(np.float32, copy=True)

    if add_self_loops:
        a += np.eye(a.shape[0], dtype=np.float32)

    if normalization == "none":
        return a

    degree = np.sum(a, axis=1)
    degree = np.clip(degree, 1e-12, None)

    if normalization == "left":
        inv_degree = 1.0 / degree
        return (inv_degree[:, None] * a).astype(np.float32)

    if normalization == "symmetric":
        inv_sqrt_degree = 1.0 / np.sqrt(degree)
        return (inv_sqrt_degree[:, None] * a * inv_sqrt_degree[None, :]).astype(np.float32)

    raise ValueError("normalization must be one of: symmetric, left, none")

def _build_component_graph(
    component_nodes: List[Any],
    node_attrs: Dict[Any, Dict[str, Any]],
    adjacency_full: Dict[Any, List[Any]],
    feats: np.ndarray,
    id_map: Dict[Any, Any],
    class_map: Dict[Any, Any],
    graph_index: int,
    normalization: str,
) -> GraphDict:
    ordered_nodes = sorted(component_nodes, key=lambda n: int(_lookup(id_map, n)))
    local_index = {node_id: idx for idx, node_id in enumerate(ordered_nodes)}

    num_nodes = len(ordered_nodes)
    first_label = _label_to_array(_lookup(class_map, ordered_nodes[0]))
    num_classes = int(first_label.shape[0])

    x = np.zeros((num_nodes, feats.shape[1]), dtype=np.float32)
    y = np.zeros((num_nodes, num_classes), dtype=np.float32)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for node_id, idx in local_index.items():
        feat_idx = int(_lookup(id_map, node_id))
        x[idx] = feats[feat_idx].astype(np.float32)
        y[idx] = _label_to_array(_lookup(class_map, node_id))

    for u in ordered_nodes:
        i = local_index[u]
        for v in adjacency_full[u]:
            if v in local_index:
                j = local_index[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    val_flags = {bool(node_attrs[n].get("val", False)) for n in ordered_nodes}
    test_flags = {bool(node_attrs[n].get("test", False)) for n in ordered_nodes}

    if len(val_flags) == 1 and True in val_flags:
        split = "val"
    elif len(test_flags) == 1 and True in test_flags:
        split = "test"
    elif len(val_flags) == 1 and False in val_flags:
        split = "train"
    else:
        raise ValueError("Connected component has mixed split flags.")

    adj_norm = normalize_adjacency(adj, add_self_loops=True, normalization=normalization)

    return {
        "name": f"graph_{graph_index:02d}_{split}",
        "split": split,
        "node_ids": ordered_nodes,
        "x": x,
        "y": y,
        "adj": adj,
        "adj_norm": adj_norm,
        "num_nodes": num_nodes,
    }


def standardize_graph_features(
    train_graphs: List[GraphDict],
    eval_graphs: List[GraphDict],
) -> Tuple[List[GraphDict], List[GraphDict], np.ndarray, np.ndarray]:
    stacked_train = np.concatenate([g["x"] for g in train_graphs], axis=0)
    mean = stacked_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = stacked_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def _apply(graphs: List[GraphDict]) -> None:
        for graph in graphs:
            graph["x"] = ((graph["x"] - mean) / std).astype(np.float32)

    _apply(train_graphs)
    _apply(eval_graphs)
    return train_graphs, eval_graphs, mean, std


def load_ppi_dataset(
    dataset_path: str,
    normalization: str = "symmetric",
    standardize_features: bool = True,
) -> Tuple[List[GraphDict], List[GraphDict], List[GraphDict], Dict[str, Any]]:
    graph_path = os.path.join(dataset_path, "ppi-G.json")
    feats_path = os.path.join(dataset_path, "ppi-feats.npy")
    class_map_path = os.path.join(dataset_path, "ppi-class_map.json")
    id_map_path = os.path.join(dataset_path, "ppi-id_map.json")

    node_ids, node_attrs, adjacency_full = load_graph_json(graph_path)
    feats = np.load(feats_path).astype(np.float32)

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    components = connected_components_manual(node_ids, adjacency_full)
    components = sorted(components, key=lambda comp: min(int(_lookup(id_map, n)) for n in comp))

    train_graphs: List[GraphDict] = []
    val_graphs: List[GraphDict] = []
    test_graphs: List[GraphDict] = []

    for graph_index, component_nodes in enumerate(components):
        graph_data = _build_component_graph(
            component_nodes=component_nodes,
            node_attrs=node_attrs,
            adjacency_full=adjacency_full,
            feats=feats,
            id_map=id_map,
            class_map=class_map,
            graph_index=graph_index,
            normalization=normalization,
        )

        if graph_data["split"] == "train":
            train_graphs.append(graph_data)
        elif graph_data["split"] == "val":
            val_graphs.append(graph_data)
        else:
            test_graphs.append(graph_data)

    if standardize_features:
        train_graphs, val_graphs, mean, std = standardize_graph_features(train_graphs, val_graphs)
        for graph in test_graphs:
            graph["x"] = ((graph["x"] - mean) / std).astype(np.float32)
    else:
        mean = np.zeros((1, feats.shape[1]), dtype=np.float32)
        std = np.ones((1, feats.shape[1]), dtype=np.float32)

    if train_graphs:
        output_dim = train_graphs[0]["y"].shape[1]
    elif val_graphs:
        output_dim = val_graphs[0]["y"].shape[1]
    elif test_graphs:
        output_dim = test_graphs[0]["y"].shape[1]
    else:
        raise ValueError("No train/val/test graphs were found in the dataset.")

    meta = {
        "input_dim": feats.shape[1],
        "output_dim": output_dim,
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "num_test_graphs": len(test_graphs),
        "feature_mean": mean,
        "feature_std": std,
        "normalization": normalization,
    }

    return train_graphs, val_graphs, test_graphs, meta

# File saving helpers
def save_history_json(history: Dict[str, List[float]], path: str) -> None:
    serializable = {k: [float(v) for v in values] for k, values in history.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def compute_pos_weight(
    train_graphs: List[GraphDict],
    max_pos_weight: float,
    pos_weight_power: float,
) -> np.ndarray:
    all_targets = np.concatenate([graph["y"] for graph in train_graphs], axis=0).astype(np.float32)
    pos_counts = np.sum(all_targets, axis=0, keepdims=True)
    neg_counts = all_targets.shape[0] - pos_counts
    raw_pos_weight = neg_counts / np.clip(pos_counts, 1.0, None)
    softened_pos_weight = np.power(np.clip(raw_pos_weight, 1.0, None), pos_weight_power)
    pos_weight = np.clip(softened_pos_weight, 1.0, max_pos_weight).astype(np.float32)
    return pos_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate a NumPy GCN on the PPI dataset.")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the PPI dataset folder")

    parser.add_argument(
        "--model_save_path",
        type=str,
        default="gcn_ppi_model.npz",
        help="Exact path where the trained model weights will be saved",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved .npz model for evaluation mode",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Directory where logs and plots will be saved. Default: outputs/<experiment_name>",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
        help="Which split to evaluate in --mode eval",
    )

    parser.add_argument("--epochs", type=int, default=140, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of GCN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for hidden layers")
    parser.add_argument("--input_dropout", type=float, default=0.0, help="Dropout rate applied to node features before the first GCN layer")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "leaky_relu"], help="Hidden-layer activation")
    parser.add_argument("--negative_slope", type=float, default=0.05, help="Negative slope for leaky ReLU")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay")
    parser.add_argument(
        "--normalization",
        type=str,
        default="symmetric",
        choices=["symmetric", "left", "none"],
        help="Adjacency normalization type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument(
        "--no_feature_standardization",
        action="store_true",
        help="Disable train-set feature standardization during training",
    )
    parser.add_argument(
        "--disable_pos_weight",
        action="store_true",
        help="Disable class-imbalance positive weighting in BCE loss",
    )
    parser.add_argument(
        "--max_pos_weight",
        type=float,
        default=8.0,
        help="Maximum clipping value for per-label positive weight after softening",
    )
    parser.add_argument(
        "--pos_weight_power",
        type=float,
        default=0.5,
        help="Softening exponent for per-label positive weights. 0.5 means square-root scaling.",
    )
    parser.add_argument(
        "--disable_residual",
        action="store_true",
        help="Disable residual connections between same-width hidden layers",
    )
    parser.add_argument(
        "--grad_clip_value",
        type=float,
        default=5.0,
        help="Elementwise gradient clipping value for Adam updates; use negative to disable",
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.5,
        help="Multiply learning rate by this factor when validation Micro-F1 plateaus",
    )
    parser.add_argument(
        "--lr_decay_patience",
        type=int,
        default=12,
        help="Number of epochs without validation improvement before decaying the learning rate",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=5e-4,
        help="Minimum learning rate after decay",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def get_experiment_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name if name else "experiment"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def prepare_artifact_paths(model_reference_path: str, artifacts_dir: Optional[str]) -> Dict[str, str]:
    experiment_name = get_experiment_name_from_path(model_reference_path)

    if artifacts_dir is None:
        artifacts_dir = os.path.join("outputs", experiment_name)

    logs_dir = os.path.join(artifacts_dir, "logs")
    plots_dir = os.path.join(artifacts_dir, "plots")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return {
        "experiment_name": experiment_name,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "history_path": os.path.join(logs_dir, "training_history.json"),
        "config_path": os.path.join(logs_dir, "run_config.json"),
        "summary_path": os.path.join(logs_dir, "summary.json"),
        "loss_plot_path": os.path.join(plots_dir, "loss_curve.png"),
        "f1_plot_path": os.path.join(plots_dir, "micro_f1_curve.png"),
    }


def apply_saved_standardization(graphs: List[GraphDict], mean: np.ndarray, std: np.ndarray) -> None:
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    for graph in graphs:
        graph["x"] = ((graph["x"] - mean) / std).astype(np.float32)


def evaluate_split(
    graphs: List[GraphDict],
    model: GCNModel,
) -> Dict[str, float]:
    losses = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for graph in graphs:
        logits = model.forward(graph["x"], graph["adj_norm"], training=False)
        loss = binary_cross_entropy_with_logits(logits, graph["y"], pos_weight=None)
        stats = micro_f1_stats_from_logits(logits, graph["y"], threshold=0.5)

        losses.append(loss)
        total_tp += stats["tp"]
        total_fp += stats["fp"]
        total_fn += stats["fn"]

    mean_loss = float(np.mean(losses)) if losses else 0.0
    micro_f1 = micro_f1_from_stats(total_tp, total_fp, total_fn)
    return {"loss": mean_loss, "micro_f1": micro_f1}


def save_metric_plots(history: Dict[str, List[float]], loss_plot_path: str, f1_plot_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_micro_f1"], label="train_micro_f1")
    plt.plot(history["val_micro_f1"], label="val_micro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Micro-F1")
    plt.title("Training and Validation Micro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f1_plot_path)
    plt.close()


def save_run_config(
    args: argparse.Namespace,
    meta: Dict[str, Any],
    experiment_name: str,
    best_val_f1: float,
    best_epoch: int,
    config_path: str,
    pos_weight: Optional[np.ndarray],
    final_lr: float,
) -> None:
    config = {
        "experiment_name": experiment_name,
        "mode": args.mode,
        "dataset_path": args.dataset_path,
        "model_save_path": args.model_save_path,
        "model_path": args.model_path,
        "artifacts_dir": args.artifacts_dir,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_lr": final_lr,
        "lr_decay_factor": args.lr_decay_factor,
        "lr_decay_patience": args.lr_decay_patience,
        "min_lr": args.min_lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "input_dropout": args.input_dropout,
        "activation": args.activation,
        "negative_slope": args.negative_slope,
        "weight_decay": args.weight_decay,
        "normalization": args.normalization,
        "seed": args.seed,
        "patience": args.patience,
        "feature_standardization": not args.no_feature_standardization,
        "batch_size": 1,
        "use_pos_weight": not args.disable_pos_weight,
        "max_pos_weight": args.max_pos_weight,
        "pos_weight_power": args.pos_weight_power,
        "use_residual": not args.disable_residual,
        "grad_clip_value": args.grad_clip_value,
        "input_dim": meta["input_dim"],
        "output_dim": meta["output_dim"],
        "num_train_graphs": meta["num_train_graphs"],
        "num_val_graphs": meta["num_val_graphs"],
        "num_test_graphs": meta["num_test_graphs"],
        "best_val_micro_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
    }
    if pos_weight is not None:
        config["pos_weight_min"] = float(np.min(pos_weight))
        config["pos_weight_max"] = float(np.max(pos_weight))
        config["pos_weight_mean"] = float(np.mean(pos_weight))

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def save_summary(
    experiment_name: str,
    best_val_f1: float,
    best_epoch: int,
    final_train_loss: float,
    final_val_loss: float,
    final_train_f1: float,
    final_val_f1: float,
    summary_path: str,
) -> None:
    summary = {
        "experiment_name": experiment_name,
        "best_val_micro_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "final_train_micro_f1": float(final_train_f1),
        "final_val_micro_f1": float(final_val_f1),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def train_mode(args: argparse.Namespace) -> None:
    ensure_parent_dir(args.model_save_path)
    artifact_paths = prepare_artifact_paths(args.model_save_path, args.artifacts_dir)

    train_graphs, val_graphs, test_graphs, meta = load_ppi_dataset(
        dataset_path=args.dataset_path,
        normalization=args.normalization,
        standardize_features=not args.no_feature_standardization,
    )

    pos_weight = None
    if not args.disable_pos_weight:
        pos_weight = compute_pos_weight(
            train_graphs,
            max_pos_weight=args.max_pos_weight,
            pos_weight_power=args.pos_weight_power,
        )

    grad_clip_value = None if args.grad_clip_value is not None and args.grad_clip_value < 0 else args.grad_clip_value

    print("Loaded dataset")
    print(f"  Experiment:         {artifact_paths['experiment_name']}")
    print(f"  Train graphs:       {meta['num_train_graphs']}")
    print(f"  Val graphs:         {meta['num_val_graphs']}")
    print(f"  Test graphs:        {meta['num_test_graphs']}")
    print(f"  Input dim:          {meta['input_dim']}")
    print(f"  Output dim:         {meta['output_dim']}")
    print(f"  Normalization:      {meta['normalization']}")
    print(f"  Residual:           {not args.disable_residual}")
    print(f"  Hidden dropout:     {args.dropout}")
    print(f"  Input dropout:      {args.input_dropout}")
    print(f"  Activation:         {args.activation}")
    if args.activation == "leaky_relu":
        print(f"  Negative slope:     {args.negative_slope}")
    print(f"  Pos weight BCE:     {not args.disable_pos_weight}")
    if pos_weight is not None:
        print(f"  Pos weight min/max/mean: {np.min(pos_weight):.2f} / {np.max(pos_weight):.2f} / {np.mean(pos_weight):.2f}")
    print(f"  LR decay factor:    {args.lr_decay_factor}")
    print(f"  LR decay patience:  {args.lr_decay_patience}")
    print(f"  Model path:         {args.model_save_path}")

    _ = test_graphs

    model = GCNModel(
        input_dim=meta["input_dim"],
        hidden_dim=args.hidden_dim,
        output_dim=meta["output_dim"],
        num_layers=args.num_layers,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        seed=args.seed,
        use_residual=not args.disable_residual,
        grad_clip_value=grad_clip_value,
        activation=args.activation,
        negative_slope=args.negative_slope,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_micro_f1": [],
        "val_micro_f1": [],
        "learning_rate": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    epochs_since_lr_decay_improvement = 0
    current_lr = float(args.lr)

    for epoch in range(1, args.epochs + 1):
        shuffled_indices = np.random.permutation(len(train_graphs))
        epoch_train_losses = []

        for idx in shuffled_indices:
            graph = train_graphs[int(idx)]
            logits = model.forward(graph["x"], graph["adj_norm"], training=True)
            loss = binary_cross_entropy_with_logits(logits, graph["y"], pos_weight=pos_weight)
            grad = bce_with_logits_gradient(logits, graph["y"], pos_weight=pos_weight)

            model.zero_grad()
            model.backward(grad)
            model.step(lr=current_lr, weight_decay=args.weight_decay)
            epoch_train_losses.append(loss)

        train_metrics = evaluate_split(train_graphs, model)
        val_metrics = evaluate_split(val_graphs, model)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_micro_f1"].append(train_metrics["micro_f1"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.6f} | "
            f"batch_train_loss={np.mean(epoch_train_losses):.4f} | "
            f"train_loss={train_metrics['loss']:.4f} | train_f1={train_metrics['micro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['micro_f1']:.4f}"
        )

        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            epochs_since_lr_decay_improvement = 0

            payload = model.state_dict(
                feature_mean=meta["feature_mean"],
                feature_std=meta["feature_std"],
                normalization=args.normalization,
            )
            if pos_weight is not None:
                payload["pos_weight"] = pos_weight.astype(np.float32)
            np.savez(args.model_save_path, **payload)
        else:
            epochs_without_improvement += 1
            epochs_since_lr_decay_improvement += 1

        if (
            args.lr_decay_factor < 1.0
            and epochs_since_lr_decay_improvement >= args.lr_decay_patience
            and current_lr > args.min_lr + 1e-12
        ):
            new_lr = max(args.min_lr, current_lr * args.lr_decay_factor)
            if new_lr < current_lr - 1e-12:
                print(f"Learning rate decay: {current_lr:.6f} -> {new_lr:.6f}")
                current_lr = new_lr
            epochs_since_lr_decay_improvement = 0

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    save_history_json(history, artifact_paths["history_path"])
    save_metric_plots(history, artifact_paths["loss_plot_path"], artifact_paths["f1_plot_path"])
    save_run_config(
        args=args,
        meta=meta,
        experiment_name=artifact_paths["experiment_name"],
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        config_path=artifact_paths["config_path"],
        pos_weight=pos_weight,
        final_lr=current_lr,
    )
    save_summary(
        experiment_name=artifact_paths["experiment_name"],
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        final_train_loss=history["train_loss"][-1],
        final_val_loss=history["val_loss"][-1],
        final_train_f1=history["train_micro_f1"][-1],
        final_val_f1=history["val_micro_f1"][-1],
        summary_path=artifact_paths["summary_path"],
    )

    print("\nTraining complete.")
    print(f"Best validation Micro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Final learning rate:      {current_lr:.6f}")
    print(f"Model saved to: {args.model_save_path}")
    print(f"Artifacts dir:  {artifact_paths['artifacts_dir']}")


def eval_mode(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise ValueError("--model_path is required when --mode eval")

    artifact_paths = prepare_artifact_paths(args.model_path, args.artifacts_dir)

    data = np.load(args.model_path, allow_pickle=True)
    saved_normalization = str(data["normalization"].item()) if "normalization" in data else args.normalization
    feature_mean = data["feature_mean"] if "feature_mean" in data else None
    feature_std = data["feature_std"] if "feature_std" in data else None

    train_graphs, val_graphs, test_graphs, meta = load_ppi_dataset(
        dataset_path=args.dataset_path,
        normalization=saved_normalization,
        standardize_features=False,
    )

    if feature_mean is not None and feature_std is not None:
        apply_saved_standardization(train_graphs, feature_mean, feature_std)
        apply_saved_standardization(val_graphs, feature_mean, feature_std)
        apply_saved_standardization(test_graphs, feature_mean, feature_std)

    model = GCNModel.from_npz(args.model_path)

    split_map = {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }

    if args.eval_split == "all":
        results: Dict[str, Dict[str, float]] = {}
        for split_name, graphs in split_map.items():
            if graphs:
                results[split_name] = evaluate_split(graphs, model)

        summary = {
            "mode": "eval",
            "model_path": args.model_path,
            "normalization": saved_normalization,
            "num_train_graphs": meta["num_train_graphs"],
            "num_val_graphs": meta["num_val_graphs"],
            "num_test_graphs": meta["num_test_graphs"],
            "results": results,
        }

        with open(artifact_paths["summary_path"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("Evaluation complete.")
        print(f"Model path:    {args.model_path}")
        for split_name, metrics in results.items():
            print(f"{split_name.capitalize()} loss:      {metrics['loss']:.4f}")
            print(f"{split_name.capitalize()} Micro-F1:  {metrics['micro_f1']:.4f}")
        print(f"Summary saved: {artifact_paths['summary_path']}")
        return

    graphs = split_map[args.eval_split]
    if not graphs:
        raise ValueError(f"No graphs found for split: {args.eval_split}")

    metrics = evaluate_split(graphs, model)
    summary = {
        "mode": "eval",
        "model_path": args.model_path,
        "eval_split": args.eval_split,
        "normalization": saved_normalization,
        "loss": float(metrics["loss"]),
        "micro_f1": float(metrics["micro_f1"]),
    }
    with open(artifact_paths["summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete.")
    print(f"Model path:    {args.model_path}")
    print(f"Eval split:    {args.eval_split}")
    print(f"Loss:          {metrics['loss']:.4f}")
    print(f"Micro-F1:      {metrics['micro_f1']:.4f}")
    print(f"Summary saved: {artifact_paths['summary_path']}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        train_mode(args)
    else:
        eval_mode(args)


if __name__ == "__main__":
    main()
