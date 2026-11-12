"""Experiment 9: stochastic GAS estimands in a real PyTorch network.

This experiment transfers the estimand distinction established analytically in
Experiment 8 to a trained Monte Carlo-dropout neural network on a real dataset.
The network is treated as a black box after training.  Explicit dropout masks
allow baseline and coordinate-replacement evaluations to share exactly the
same stochastic realization (common random numbers, CRN).

The numerical targets are

    C_path = E[Q Q^T]
    C_mu   = E[E(Q | X,V) E(Q | X,V)^T],

where Q is the vector of coordinate-replacement secants.  Unlike Experiment 8,
these matrices are high-budget numerical references, not analytic truths.

Protocols:

1. one-mask CRN, targeting C_path;
2. K-mask response averaging, targeting C_mu + (C_path-C_mu)/K;
3. cross-mask U-statistic, targeting C_mu;
4. independent-mask finite differences, a negative control.

The default ``smoke`` profile is intentionally small.  It validates the
implementation and expected qualitative trends before a larger multi-network
study is run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "pytorch_stochastic_estimands_smoke"
EPS = np.finfo(float).eps


@dataclass(frozen=True)
class Profile:
    training_seeds: tuple[int, ...]
    epochs: int
    anchors: int
    reference_masks: int
    repeats: int
    k_values: tuple[int, ...]


PROFILES = {
    "smoke": Profile(
        training_seeds=(1101, 2202),
        epochs=120,
        anchors=64,
        reference_masks=64,
        repeats=4,
        k_values=(1, 2, 4, 8),
    ),
    "full": Profile(
        training_seeds=(1101, 2202, 3303, 4404, 5505),
        epochs=250,
        anchors=96,
        reference_masks=128,
        repeats=12,
        k_values=(1, 2, 4, 8, 16),
    ),
}


def seeded_rng(base_seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base_seed, *parts]))


class FixedMaskDropoutMLP(nn.Module):
    """Two-hidden-layer MLP with externally controllable dropout masks."""

    def __init__(self, input_dim: int, dropout_rate: float = 0.20) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.dropout_rate = float(dropout_rate)

    @property
    def keep_probability(self) -> float:
        return 1.0 - self.dropout_rate

    def _drop(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None,
        stochastic: bool,
    ) -> torch.Tensor:
        if not stochastic:
            return values
        if mask is None:
            mask = torch.bernoulli(torch.full_like(values, self.keep_probability))
            mask = mask / self.keep_probability
        if mask.shape != values.shape:
            raise ValueError(f"mask shape {mask.shape} does not match {values.shape}")
        return values * mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        stochastic: bool = False,
        mask1: torch.Tensor | None = None,
        mask2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden1 = torch.relu(self.hidden1(x))
        hidden1 = self._drop(hidden1, mask1, stochastic)
        hidden2 = torch.relu(self.hidden2(hidden1))
        hidden2 = self._drop(hidden2, mask2, stochastic)
        return self.output(hidden2).squeeze(-1)


def explicit_masks(
    rows: int,
    width: int,
    keep_probability: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    mask = rng.binomial(1, keep_probability, size=(rows, width)).astype(np.float32)
    return torch.from_numpy(mask / keep_probability)


def load_regression_data(split_seed: int = 20261111) -> dict[str, np.ndarray]:
    dataset = load_diabetes()
    x = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.float32)
    x_build, x_test, y_build, y_test = train_test_split(
        x, y, test_size=0.20, random_state=split_seed
    )
    x_train_validation, x_sensitivity, y_train_validation, y_sensitivity = train_test_split(
        x_build, y_build, test_size=0.20, random_state=split_seed + 1
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.125,
        random_state=split_seed + 2,
    )

    x_scaler = StandardScaler().fit(x_train)
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std(ddof=0))
    if y_scale <= 0.0:
        raise ValueError("training response has zero variance")

    return {
        "x_train": x_scaler.transform(x_train).astype(np.float32),
        "x_validation": x_scaler.transform(x_validation).astype(np.float32),
        "x_sensitivity": x_scaler.transform(x_sensitivity).astype(np.float32),
        "x_test": x_scaler.transform(x_test).astype(np.float32),
        "y_train": ((y_train - y_mean) / y_scale).astype(np.float32),
        "y_validation": ((y_validation - y_mean) / y_scale).astype(np.float32),
        "y_sensitivity": ((y_sensitivity - y_mean) / y_scale).astype(np.float32),
        "y_test": ((y_test - y_mean) / y_scale).astype(np.float32),
        "y_mean": np.array(y_mean),
        "y_scale": np.array(y_scale),
        "feature_names": np.asarray(dataset.feature_names),
    }


def train_model(
    data: dict[str, np.ndarray],
    seed: int,
    epochs: int,
    dropout_rate: float,
    batch_size: int = 32,
    early_stopping_patience: int = 30,
) -> tuple[FixedMaskDropoutMLP, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FixedMaskDropoutMLP(data["x_train"].shape[1], dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed + 91_337)
    dataset = TensorDataset(
        torch.from_numpy(data["x_train"]), torch.from_numpy(data["y_train"])
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    start = time.perf_counter()
    model.train()
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_completed = 0
    validation_x = torch.from_numpy(data["x_validation"])
    validation_y = torch.from_numpy(data["y_validation"])
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch, stochastic=True)
            loss = loss_function(prediction, y_batch)
            loss.backward()
            optimizer.step()
        epochs_completed = epoch + 1
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x)
            validation_loss = float(loss_function(validation_prediction, validation_y))
        model.train()
        if validation_loss < best_validation_loss - 1.0e-5:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            break
    train_seconds = time.perf_counter() - start

    if best_state is None:
        raise AssertionError("early stopping did not save a model state")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_prediction = model(torch.from_numpy(data["x_train"])).numpy()
        validation_prediction = model(torch.from_numpy(data["x_validation"])).numpy()
        test_prediction = model(torch.from_numpy(data["x_test"])).numpy()

    metrics = {
        "training_seed": int(seed),
        "epochs": int(epochs),
        "epochs_completed": int(epochs_completed),
        "best_epoch": int(best_epoch),
        "train_seconds": float(train_seconds),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_r2": float(r2_score(data["y_train"], train_prediction)),
        "validation_r2": float(
            r2_score(data["y_validation"], validation_prediction)
        ),
        "test_r2": float(r2_score(data["y_test"], test_prediction)),
        "train_rmse_standardized": float(
            math.sqrt(mean_squared_error(data["y_train"], train_prediction))
        ),
        "test_rmse_standardized": float(
            math.sqrt(mean_squared_error(data["y_test"], test_prediction))
        ),
        "test_rmse_original_units": float(
            math.sqrt(mean_squared_error(data["y_test"], test_prediction))
            * float(data["y_scale"])
        ),
    }
    return model, metrics


def replacement_design(
    sensitivity_inputs: np.ndarray,
    replacement_pool: np.ndarray,
    anchors: int,
    min_separation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if anchors > len(sensitivity_inputs):
        anchor_indices = rng.choice(len(sensitivity_inputs), size=anchors, replace=True)
    else:
        anchor_indices = rng.choice(len(sensitivity_inputs), size=anchors, replace=False)
    x = sensitivity_inputs[anchor_indices].copy()
    v = np.empty_like(x)
    for i in range(anchors):
        for j in range(x.shape[1]):
            candidates = replacement_pool[
                np.abs(replacement_pool[:, j] - x[i, j]) >= min_separation, j
            ]
            if len(candidates) == 0:
                raise ValueError(f"no valid replacements for anchor={i}, coordinate={j}")
            v[i, j] = rng.choice(candidates)
    return x, v


@torch.no_grad()
def coupled_secants(
    model: FixedMaskDropoutMLP,
    x: np.ndarray,
    v: np.ndarray,
    masks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return CRN secants with shape (anchors, masks, dimension)."""
    n, d = x.shape
    rows = n * masks
    x_repeated = np.repeat(x, masks, axis=0)
    mask1 = explicit_masks(rows, 64, model.keep_probability, rng)
    mask2 = explicit_masks(rows, 32, model.keep_probability, rng)
    baseline = model(
        torch.from_numpy(x_repeated), stochastic=True, mask1=mask1, mask2=mask2
    ).numpy()
    q = np.empty((n, masks, d), dtype=np.float64)
    delta = v - x
    for j in range(d):
        perturbed = x_repeated.copy()
        perturbed[:, j] = np.repeat(v[:, j], masks)
        response = model(
            torch.from_numpy(perturbed), stochastic=True, mask1=mask1, mask2=mask2
        ).numpy()
        q[:, :, j] = ((response - baseline) / np.repeat(delta[:, j], masks)).reshape(
            n, masks
        )
    return q


@torch.no_grad()
def independent_mask_secants(
    model: FixedMaskDropoutMLP,
    x: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return secants whose baseline and replacement use unrelated masks."""
    n, d = x.shape
    base_inputs = np.repeat(x, d, axis=0)
    perturbed_inputs = base_inputs.copy()
    coordinates = np.tile(np.arange(d), n)
    repeated_v = np.repeat(v, d, axis=0)
    perturbed_inputs[np.arange(n * d), coordinates] = repeated_v[
        np.arange(n * d), coordinates
    ]
    baseline = model(
        torch.from_numpy(base_inputs),
        stochastic=True,
        mask1=explicit_masks(n * d, 64, model.keep_probability, rng),
        mask2=explicit_masks(n * d, 32, model.keep_probability, rng),
    ).numpy()
    perturbed = model(
        torch.from_numpy(perturbed_inputs),
        stochastic=True,
        mask1=explicit_masks(n * d, 64, model.keep_probability, rng),
        mask2=explicit_masks(n * d, 32, model.keep_probability, rng),
    ).numpy()
    delta = (v - x).reshape(-1)
    return ((perturbed - baseline) / delta).reshape(n, d)


def outer_mean(vectors: np.ndarray) -> np.ndarray:
    flattened = np.asarray(vectors, dtype=float).reshape(-1, vectors.shape[-1])
    matrix = np.einsum("nd,ne->de", flattened, flattened) / len(flattened)
    return 0.5 * (matrix + matrix.T)


def k_average_matrix(q: np.ndarray, k: int) -> np.ndarray:
    return outer_mean(q[:, :k].mean(axis=1))


def cross_mask_u_statistic(q: np.ndarray, k: int) -> np.ndarray:
    if k < 2:
        raise ValueError("cross-mask U-statistic requires k >= 2")
    selected = q[:, :k]
    summed = selected.sum(axis=1)
    outer_sum = np.einsum("nd,ne->nde", summed, summed)
    diagonal_sum = np.einsum("nkd,nke->nde", selected, selected)
    matrix = ((outer_sum - diagonal_sum) / (k * (k - 1))).mean(axis=0)
    return 0.5 * (matrix + matrix.T)


def eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def project_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = eigensystem(matrix)
    return (vectors * np.maximum(values, 0.0)) @ vectors.T


def effective_dimension(matrix: np.ndarray, fraction: float = 0.90) -> int:
    values, _ = eigensystem(matrix)
    positive = np.maximum(values, 0.0)
    if positive.sum() <= 0.0:
        return len(values)
    return int(np.searchsorted(np.cumsum(positive) / positive.sum(), fraction) + 1)


def leading_axis_cosine(estimate: np.ndarray, target: np.ndarray) -> float:
    _, estimate_vectors = eigensystem(estimate)
    _, target_vectors = eigensystem(target)
    return float(abs(estimate_vectors[:, 0] @ target_vectors[:, 0]))


def subspace_overlap(estimate: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    rank = effective_dimension(target)
    _, estimate_vectors = eigensystem(project_psd(estimate))
    _, target_vectors = eigensystem(target)
    singular_values = np.linalg.svd(
        estimate_vectors[:, :rank].T @ target_vectors[:, :rank], compute_uv=False
    )
    return float(np.mean(singular_values**2)), rank


def relative_frobenius(estimate: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - target, ord="fro")
        / max(np.linalg.norm(target, ord="fro"), EPS)
    )


def matrix_metrics(
    estimate: np.ndarray,
    c_mu: np.ndarray,
    c_path: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    estimate = 0.5 * (estimate + estimate.T)
    if not np.all(np.isfinite(estimate)):
        raise FloatingPointError("non-finite GAS matrix")
    values, _ = eigensystem(estimate)
    overlap, target_rank = subspace_overlap(estimate, target)
    return {
        "fro_rel_mu": relative_frobenius(estimate, c_mu),
        "fro_rel_path": relative_frobenius(estimate, c_path),
        "fro_rel_target": relative_frobenius(estimate, target),
        "leading_cos_mu": leading_axis_cosine(estimate, c_mu),
        "leading_cos_path": leading_axis_cosine(estimate, c_path),
        "leading_cos_target": leading_axis_cosine(estimate, target),
        "target_subspace_overlap": overlap,
        "target_effective_dimension_90": float(target_rank),
        "estimate_effective_dimension_90": float(effective_dimension(estimate)),
        "trace": float(np.trace(estimate)),
        "min_eigenvalue": float(values.min()),
        "negative_eigenvalue_mass": float(np.maximum(-values, 0.0).sum()),
    }


def add_row(
    rows: list[dict[str, object]],
    *,
    seed: int,
    repeat: int,
    protocol: str,
    target_label: str,
    k: int,
    estimate: np.ndarray,
    c_mu: np.ndarray,
    c_path: np.ndarray,
    target: np.ndarray,
    logical_evaluations: int,
    seconds: float,
) -> None:
    row: dict[str, object] = {
        "training_seed": seed,
        "repeat": repeat,
        "protocol": protocol,
        "target_label": target_label,
        "k": k,
        "logical_evaluations": logical_evaluations,
        "estimation_seconds": seconds,
    }
    row.update(matrix_metrics(estimate, c_mu, c_path, target))
    rows.append(row)


METRICS = [
    "fro_rel_mu",
    "fro_rel_path",
    "fro_rel_target",
    "leading_cos_mu",
    "leading_cos_path",
    "leading_cos_target",
    "target_subspace_overlap",
    "target_effective_dimension_90",
    "estimate_effective_dimension_90",
    "negative_eigenvalue_mass",
    "estimation_seconds",
]


def model_level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["training_seed", "protocol", "target_label", "k", "logical_evaluations"]
    return frame.groupby(groups, as_index=False)[METRICS].mean()


def across_model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = ["protocol", "target_label", "k", "logical_evaluations"]
    rows: list[dict[str, object]] = []
    for key, subset in frame.groupby(groups, sort=True):
        row = dict(zip(groups, key))
        row["networks"] = int(len(subset))
        for metric in METRICS:
            values = subset[metric].to_numpy(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            critical = float(student_t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
            half_width = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrasts(model_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    maximum_k = int(model_summary.k.max())
    for training_seed, subset in model_summary.groupby("training_seed"):
        def value(protocol: str, k: int, metric: str) -> float:
            selected = subset[(subset.protocol == protocol) & (subset.k == k)]
            if len(selected) != 1:
                raise AssertionError(f"missing {protocol}, K={k}, seed={training_seed}")
            return float(selected.iloc[0][metric])

        rows.extend(
            [
                {
                    "training_seed": training_seed,
                    "contrast": "independent_minus_paired_path_error",
                    "value": value("independent_masks", 1, "fro_rel_path")
                    - value("one_mask_crn", 1, "fro_rel_path"),
                    "positive_means": "independent masks have larger C_path error",
                },
                {
                    "training_seed": training_seed,
                    "contrast": "paired_minus_independent_target_cosine",
                    "value": value("one_mask_crn", 1, "leading_cos_target")
                    - value("independent_masks", 1, "leading_cos_target"),
                    "positive_means": "paired masks recover the target axis better",
                },
                {
                    "training_seed": training_seed,
                    "contrast": f"k_average_K1_minus_K{maximum_k}_mu_error",
                    "value": value("k_mask_average", 1, "fro_rel_mu")
                    - value("k_mask_average", maximum_k, "fro_rel_mu"),
                    "positive_means": "averaging more masks reduces C_mu error",
                },
            ]
        )
        for k in sorted(model_summary[model_summary.protocol == "cross_mask_u_stat"].k.unique()):
            rows.append(
                {
                    "training_seed": training_seed,
                    "contrast": f"k_average_minus_u_stat_mu_error_K{k}",
                    "value": value("k_mask_average", int(k), "fro_rel_mu")
                    - value("cross_mask_u_stat", int(k), "fro_rel_mu"),
                    "positive_means": "U-statistic has smaller C_mu error",
                }
            )

    model_frame = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, object]] = []
    for contrast, subset in model_frame.groupby("contrast", sort=True):
        values = subset.value.to_numpy(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = float(student_t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
        aggregate_rows.append(
            {
                "contrast": contrast,
                "networks": len(values),
                "mean": mean,
                "sd": sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "positive_means": str(subset.positive_means.iloc[0]),
            }
        )
    return model_frame, pd.DataFrame(aggregate_rows)


def save_summary_figure(
    summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    path: Path,
    profile_name: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    k_average = summary[summary.protocol == "k_mask_average"].sort_values("k")
    u_stat = summary[summary.protocol == "cross_mask_u_stat"].sort_values("k")
    axes[0].plot(k_average.k, k_average.fro_rel_mu_mean, "o-", label="K-mask average")
    axes[0].plot(u_stat.k, u_stat.fro_rel_mu_mean, "s-", label="Cross-mask U-stat")
    axes[0].set(
        xlabel="K masks per anchor",
        ylabel=r"Relative error to numerical $C_\mu$",
        title="Mean-response target",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    selected = summary[
        ((summary.protocol == "one_mask_crn") & (summary.k == 1))
        | ((summary.protocol == "independent_masks") & (summary.k == 1))
    ]
    axes[1].bar(
        ["Paired CRN", "Independent"],
        [
            float(selected[selected.protocol == "one_mask_crn"].fro_rel_path_mean.iloc[0]),
            float(selected[selected.protocol == "independent_masks"].fro_rel_path_mean.iloc[0]),
        ],
        color=["#4C78A8", "#E45756"],
    )
    axes[1].set(
        ylabel=r"Relative error to numerical $C_{\rm path}$",
        title="Why masks are paired",
    )
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        [str(seed) for seed in model_metrics.training_seed],
        model_metrics.test_r2,
        color="#59A14F",
    )
    axes[2].set(xlabel="Training seed", ylabel="Held-out R²", title="Network predictive fit")
    axes[2].grid(axis="y", alpha=0.25)
    profile_label = "smoke test" if profile_name == "smoke" else "five-network study"
    fig.suptitle(
        f"Experiment 9 {profile_label}: real-data PyTorch MC-dropout GAS", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    profile_name: str,
    out: Path,
    base_seed: int,
    dropout_rate: float,
    min_separation: float,
) -> None:
    profile = PROFILES[profile_name]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    data = load_regression_data()
    n, d = profile.anchors, data["x_train"].shape[1]
    design_rng = seeded_rng(base_seed, 10)
    x, v = replacement_design(
        data["x_sensitivity"], data["x_train"], n, min_separation, design_rng
    )

    all_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, float]] = []
    spectrum_rows: list[dict[str, object]] = []
    total_start = time.perf_counter()

    for seed_index, training_seed in enumerate(profile.training_seeds):
        print(f"training network seed={training_seed}", flush=True)
        model, fit_metrics = train_model(
            data, training_seed, profile.epochs, dropout_rate
        )
        model_rows.append(fit_metrics)

        parameter_snapshot = [parameter.detach().clone() for parameter in model.parameters()]
        reference_start = time.perf_counter()
        q_reference = coupled_secants(
            model,
            x,
            v,
            profile.reference_masks,
            seeded_rng(base_seed, 20, seed_index),
        )
        reference_seconds = time.perf_counter() - reference_start
        c_path = outer_mean(q_reference)
        c_mu = cross_mask_u_statistic(q_reference, profile.reference_masks)
        c_rand = c_path - c_mu

        for target_label, matrix in [("C_mu", c_mu), ("C_path", c_path), ("C_rand_numerical", c_rand)]:
            values, vectors = eigensystem(matrix)
            for eigen_index, eigenvalue in enumerate(values):
                spectrum_rows.append(
                    {
                        "training_seed": training_seed,
                        "target": target_label,
                        "eigen_index": eigen_index + 1,
                        "eigenvalue": float(eigenvalue),
                        "leading_loading_feature": str(
                            data["feature_names"][np.argmax(np.abs(vectors[:, eigen_index]))]
                        ),
                    }
                )

        for repeat in range(profile.repeats):
            q_start = time.perf_counter()
            q = coupled_secants(
                model,
                x,
                v,
                max(profile.k_values),
                seeded_rng(base_seed, 30, seed_index, repeat),
            )
            q_seconds = time.perf_counter() - q_start

            one_mask = outer_mean(q[:, 0, :])
            add_row(
                all_rows,
                seed=training_seed,
                repeat=repeat,
                protocol="one_mask_crn",
                target_label="C_path",
                k=1,
                estimate=one_mask,
                c_mu=c_mu,
                c_path=c_path,
                target=c_path,
                logical_evaluations=n * (d + 1),
                seconds=q_seconds / max(profile.k_values),
            )

            for k in profile.k_values:
                c_k = c_mu + c_rand / k
                estimate = k_average_matrix(q, k)
                add_row(
                    all_rows,
                    seed=training_seed,
                    repeat=repeat,
                    protocol="k_mask_average",
                    target_label="C_mu_plus_Crand_over_K",
                    k=k,
                    estimate=estimate,
                    c_mu=c_mu,
                    c_path=c_path,
                    target=c_k,
                    logical_evaluations=n * k * (d + 1),
                    seconds=q_seconds * k / max(profile.k_values),
                )
                if k >= 2:
                    estimate = cross_mask_u_statistic(q, k)
                    add_row(
                        all_rows,
                        seed=training_seed,
                        repeat=repeat,
                        protocol="cross_mask_u_stat",
                        target_label="C_mu",
                        k=k,
                        estimate=estimate,
                        c_mu=c_mu,
                        c_path=c_path,
                        target=c_mu,
                        logical_evaluations=n * k * (d + 1),
                        seconds=q_seconds * k / max(profile.k_values),
                    )

            independent_start = time.perf_counter()
            independent = outer_mean(
                independent_mask_secants(
                    model,
                    x,
                    v,
                    seeded_rng(base_seed, 40, seed_index, repeat),
                )
            )
            independent_seconds = time.perf_counter() - independent_start
            add_row(
                all_rows,
                seed=training_seed,
                repeat=repeat,
                protocol="independent_masks",
                target_label="negative_control_compared_with_C_path",
                k=1,
                estimate=independent,
                c_mu=c_mu,
                c_path=c_path,
                target=c_path,
                logical_evaluations=n * 2 * d,
                seconds=independent_seconds,
            )

        for before, after in zip(parameter_snapshot, model.parameters()):
            if not torch.equal(before, after.detach()):
                raise AssertionError("model parameters changed during GAS evaluation")
        fit_metrics["reference_masks"] = profile.reference_masks
        fit_metrics["reference_seconds"] = reference_seconds

    repeat_frame = pd.DataFrame(all_rows)
    model_frame = pd.DataFrame(model_rows)
    model_summary = model_level_summary(repeat_frame)
    across_summary = across_model_summary(model_summary)
    contrast_model, contrast_across = paired_contrasts(model_summary)
    spectrum_frame = pd.DataFrame(spectrum_rows)

    repeat_frame.to_csv(out / "estimator_metrics_by_repeat.csv", index=False)
    model_summary.to_csv(out / "estimator_metrics_model_level.csv", index=False)
    across_summary.to_csv(out / "estimator_metrics_across_models.csv", index=False)
    contrast_model.to_csv(out / "paired_contrasts_model_level.csv", index=False)
    contrast_across.to_csv(out / "paired_contrasts_across_models.csv", index=False)
    model_frame.to_csv(out / "model_metrics.csv", index=False)
    spectrum_frame.to_csv(out / "numerical_reference_spectra.csv", index=False)
    save_summary_figure(
        across_summary,
        model_frame,
        out / "pytorch_stochastic_estimands_summary.png",
        profile_name,
    )

    metadata = {
        "experiment": 9,
        "profile": profile_name,
        "dataset": "scikit-learn diabetes regression",
        "architecture": "10-64-32-1 ReLU MLP",
        "dropout_rate": dropout_rate,
        "dropout_implementation": "explicit per-sample inverted-dropout masks",
        "base_seed": base_seed,
        "min_standardized_coordinate_separation": min_separation,
        "profile_parameters": asdict(profile),
        "data_sizes": {
            "training": int(len(data["x_train"])),
            "validation": int(len(data["x_validation"])),
            "sensitivity_pool": int(len(data["x_sensitivity"])),
            "test": int(len(data["x_test"])),
        },
        "reference_status": (
            "High-budget numerical references; C_mu uses the cross-mask "
            "U-statistic and neither matrix is analytic ground truth."
        ),
        "statistical_unit": "Independently trained network; repeats are internal Monte Carlo repetitions.",
        "total_seconds": time.perf_counter() - total_start,
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (out / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nHeld-out model fit:")
    print(model_frame[["training_seed", "test_r2", "test_rmse_original_units"]].to_string(index=False))
    display_columns = [
        "protocol",
        "k",
        "fro_rel_mu_mean",
        "fro_rel_path_mean",
        "leading_cos_target_mean",
        "estimate_effective_dimension_90_mean",
    ]
    print("\nGAS estimand summary across trained networks:")
    print(across_summary[display_columns].to_string(index=False))
    print(f"\nResults written to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-seed", type=int, default=20261111)
    parser.add_argument("--dropout-rate", type=float, default=0.20)
    parser.add_argument("--min-separation", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(
        profile_name=arguments.profile,
        out=arguments.out,
        base_seed=arguments.base_seed,
        dropout_rate=arguments.dropout_rate,
        min_separation=arguments.min_separation,
    )
