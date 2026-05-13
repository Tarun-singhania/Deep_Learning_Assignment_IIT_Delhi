from typing import Dict, List, Optional

import numpy as np


class GCNLayer:
    """Single graph convolution layer implemented fully in NumPy.
    Forward:
        aggregated = A_hat @ X
        Z = aggregated @ W + b
    Backward computes gradients analytically for W, b, and X.
    Adam optimizer state is stored inside each layer.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, adj_norm: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if adj_norm.ndim != 2:
            raise ValueError("adj_norm must be 2D")
        if adj_norm.shape[0] != adj_norm.shape[1]:
            raise ValueError("adj_norm must be square")
        if adj_norm.shape[0] != x.shape[0]:
            raise ValueError("adj_norm and x must have the same number of nodes")
        if x.shape[1] != self.W.shape[0]:
            raise ValueError("Input feature dimension mismatch")

        aggregated = adj_norm @ x
        z = aggregated @ self.W + self.b

        self.cache = {
            "adj_norm": adj_norm,
            "aggregated": aggregated,
        }
        return z.astype(np.float32)

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        if dZ.ndim != 2:
            raise ValueError("dZ must be 2D")

        aggregated = self.cache["aggregated"]
        adj_norm = self.cache["adj_norm"]

        self.dW = aggregated.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        d_aggregated = dZ @ self.W.T
        dX = adj_norm.T @ d_aggregated
        return dX.astype(np.float32)

    def zero_grad(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def adam_step(
        self,
        lr: float,
        t: int,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip_value: Optional[float] = None,
    ) -> None:
        grad_w = self.dW + weight_decay * self.W
        grad_b = self.db

        if grad_clip_value is not None and grad_clip_value > 0.0:
            grad_w = np.clip(grad_w, -grad_clip_value, grad_clip_value)
            grad_b = np.clip(grad_b, -grad_clip_value, grad_clip_value)

        self.mW = beta1 * self.mW + (1.0 - beta1) * grad_w
        self.vW = beta2 * self.vW + (1.0 - beta2) * (grad_w ** 2)
        self.mb = beta1 * self.mb + (1.0 - beta1) * grad_b
        self.vb = beta2 * self.vb + (1.0 - beta2) * (grad_b ** 2)

        mW_hat = self.mW / (1.0 - beta1 ** t)
        vW_hat = self.vW / (1.0 - beta2 ** t)
        mb_hat = self.mb / (1.0 - beta1 ** t)
        vb_hat = self.vb / (1.0 - beta2 ** t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class GCNModel:
    """Stacked multi-layer GCN architecture.
    Hidden layers use configurable activation + optional residual connections + dropout.
    Final layer outputs logits for multi-label prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        input_dropout: float = 0.0,
        seed: int = 42,
        use_residual: bool = True,
        grad_clip_value: Optional[float] = 5.0,
        activation: str = "relu",
        negative_slope: float = 0.05,
    ) -> None:
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        np.random.seed(seed)

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.input_dropout = float(input_dropout)
        self.global_step = 0
        self.use_residual = bool(use_residual)
        self.grad_clip_value = None if grad_clip_value is None else float(grad_clip_value)
        self.activation = str(activation)
        self.negative_slope = float(negative_slope)
        if self.activation not in {"relu", "leaky_relu"}:
            raise ValueError("activation must be one of: relu, leaky_relu")

        dims = [self.input_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.output_dim]
        self.layers: List[GCNLayer] = [GCNLayer(dims[i], dims[i + 1]) for i in range(self.num_layers)]
        self.hidden_caches: List[Dict[str, Optional[np.ndarray]]] = []


    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0).astype(np.float32)
        return np.where(z > 0.0, z, self.negative_slope * z).astype(np.float32)

    def _activation_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        return np.where(z > 0.0, 1.0, self.negative_slope).astype(np.float32)

    def forward(self, x: np.ndarray, adj_norm: np.ndarray, training: bool = True) -> np.ndarray:
        h = x.astype(np.float32)
        self.hidden_caches = []

        if training and self.input_dropout > 0.0:
            keep_prob = 1.0 - self.input_dropout
            input_dropout_mask = (
                (np.random.rand(*h.shape) < keep_prob).astype(np.float32) / keep_prob
            )
            h = h * input_dropout_mask

        for layer_idx, layer in enumerate(self.layers):
            input_h = h
            z = layer.forward(h, adj_norm)

            if layer_idx == len(self.layers) - 1:
                h = z
                continue

            activation_grad = self._activation_grad(z)
            hidden = self._activate(z)

            residual_used = False
            if self.use_residual and hidden.shape == input_h.shape:
                hidden = hidden + input_h
                residual_used = True

            dropout_mask = None
            if training and self.dropout > 0.0:
                keep_prob = 1.0 - self.dropout
                dropout_mask = (
                    (np.random.rand(*hidden.shape) < keep_prob).astype(np.float32) / keep_prob
                )
                hidden *= dropout_mask

            self.hidden_caches.append(
                {
                    "activation_grad": activation_grad,
                    "dropout_mask": dropout_mask,
                    "residual_used": np.array(1 if residual_used else 0, dtype=np.int32),
                }
            )
            h = hidden

        return h.astype(np.float32)

    def backward(self, d_logits: np.ndarray) -> np.ndarray:
        grad = d_logits.astype(np.float32)

        for layer_idx in reversed(range(len(self.layers))):
            if layer_idx < len(self.layers) - 1:
                cache = self.hidden_caches[layer_idx]

                if cache["dropout_mask"] is not None:
                    grad = grad * cache["dropout_mask"]

                residual_grad = grad if bool(cache["residual_used"].item()) else None
                grad = grad * cache["activation_grad"]
                grad = self.layers[layer_idx].backward(grad)

                if residual_grad is not None:
                    grad = grad + residual_grad
            else:
                grad = self.layers[layer_idx].backward(grad)

        return grad.astype(np.float32)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        self.global_step += 1
        for layer in self.layers:
            layer.adam_step(
                lr=lr,
                t=self.global_step,
                weight_decay=weight_decay,
                grad_clip_value=self.grad_clip_value,
            )

    def state_dict(
        self,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        normalization: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {
            "input_dim": np.array(self.input_dim, dtype=np.int32),
            "hidden_dim": np.array(self.hidden_dim, dtype=np.int32),
            "output_dim": np.array(self.output_dim, dtype=np.int32),
            "num_layers": np.array(self.num_layers, dtype=np.int32),
            "dropout": np.array(self.dropout, dtype=np.float32),
            "input_dropout": np.array(self.input_dropout, dtype=np.float32),
            "global_step": np.array(self.global_step, dtype=np.int32),
            "use_residual": np.array(1 if self.use_residual else 0, dtype=np.int32),
            "grad_clip_value": np.array(-1.0 if self.grad_clip_value is None else self.grad_clip_value, dtype=np.float32),
            "activation": np.array(self.activation),
            "negative_slope": np.array(self.negative_slope, dtype=np.float32),
        }

        for idx, layer in enumerate(self.layers):
            state[f"layer_{idx}_W"] = layer.W
            state[f"layer_{idx}_b"] = layer.b

        if feature_mean is not None:
            state["feature_mean"] = feature_mean.astype(np.float32)
        if feature_std is not None:
            state["feature_std"] = feature_std.astype(np.float32)
        if normalization is not None:
            state["normalization"] = np.array(normalization)

        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.global_step = int(state.get("global_step", 0))
        for idx, layer in enumerate(self.layers):
            layer.W = state[f"layer_{idx}_W"].astype(np.float32)
            layer.b = state[f"layer_{idx}_b"].astype(np.float32)

    @classmethod
    def from_npz(cls, path: str) -> "GCNModel":
        data = np.load(path, allow_pickle=True)
        grad_clip_value = None
        if "grad_clip_value" in data:
            grad_clip_raw = float(data["grad_clip_value"])
            grad_clip_value = None if grad_clip_raw < 0.0 else grad_clip_raw

        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            output_dim=int(data["output_dim"]),
            num_layers=int(data["num_layers"]),
            dropout=float(data["dropout"]),
            input_dropout=float(data["input_dropout"]) if "input_dropout" in data else 0.0,
            use_residual=bool(int(data["use_residual"])) if "use_residual" in data else True,
            grad_clip_value=grad_clip_value,
            activation=str(data["activation"].item()) if "activation" in data else "relu",
            negative_slope=float(data["negative_slope"]) if "negative_slope" in data else 0.05,
        )
        model.load_state_dict(data)
        return model
