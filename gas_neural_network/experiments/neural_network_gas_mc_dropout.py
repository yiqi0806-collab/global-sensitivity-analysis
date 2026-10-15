"""Stage-2 GAS experiment with genuine neural-network stochasticity.

A ReLU MLP is trained with inverted dropout and evaluated with Monte Carlo
dropout.  Sobol total-effect indices, GAS, and DGSM are compared at similar
black-box evaluation budgets under three protocols:

1. deterministic inference (dropout disabled),
2. MC dropout with paired masks / common random numbers (CRN), and
3. MC dropout with independent masks for paired evaluations.

In addition to feature-ranking accuracy, the experiment measures whether the
low-dimensional GAS eigenspace is stable relative to a high-budget,
deterministic GAS reference.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from neural_network_gas_demo import (
    FEATURES,
    RELEVANT,
    font,
    friedman_function,
    kendall_tau,
    normalized,
    top_k_recovery,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "mc_dropout_pilot"


@dataclass
class DropoutMLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    y_mean: float
    y_std: float
    dropout_rate: float

    @property
    def hidden_width(self) -> int:
        return int(self.w1.shape[1])

    def sample_masks(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        keep = 1.0 - self.dropout_rate
        mask1 = (rng.random((n, self.hidden_width)) < keep).astype(float) / keep
        mask2 = (rng.random((n, self.hidden_width)) < keep).astype(float) / keep
        return mask1, mask2

    def predict(
        self,
        x: np.ndarray,
        masks: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        xs = (np.asarray(x, dtype=float) - 0.5) / math.sqrt(1.0 / 12.0)
        a1 = np.maximum(0.0, xs @ self.w1 + self.b1)
        if masks is not None:
            a1 = a1 * masks[0]
        a2 = np.maximum(0.0, a1 @ self.w2 + self.b2)
        if masks is not None:
            a2 = a2 * masks[1]
        standardized = (a2 @ self.w3 + self.b3).ravel()
        return standardized * self.y_std + self.y_mean


def train_dropout_mlp(
    *,
    seed: int = 20260930,
    steps: int = 3500,
    dropout_rate: float = 0.10,
) -> tuple[DropoutMLP, dict[str, float]]:
    rng = np.random.default_rng(seed)
    x_train = rng.random((6000, 10))
    x_test = rng.random((2500, 10))
    y_train = friedman_function(x_train)
    y_test = friedman_function(x_test)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    target = (y_train - y_mean) / y_std

    width = 32
    w1 = rng.normal(0.0, math.sqrt(2.0 / 10), (10, width))
    b1 = np.zeros(width)
    w2 = rng.normal(0.0, math.sqrt(2.0 / width), (width, width))
    b2 = np.zeros(width)
    w3 = rng.normal(0.0, math.sqrt(2.0 / width), (width, 1))
    b3 = np.zeros(1)
    params = [w1, b1, w2, b2, w3, b3]
    adam_m = [np.zeros_like(p) for p in params]
    adam_v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    batch_size = 256
    keep = 1.0 - dropout_rate

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(x_train), size=batch_size)
        xb = (x_train[idx] - 0.5) / math.sqrt(1.0 / 12.0)
        yb = target[idx, None]

        z1 = xb @ w1 + b1
        a1 = np.maximum(0.0, z1)
        mask1 = (rng.random(a1.shape) < keep).astype(float) / keep
        a1_drop = a1 * mask1
        z2 = a1_drop @ w2 + b2
        a2 = np.maximum(0.0, z2)
        mask2 = (rng.random(a2.shape) < keep).astype(float) / keep
        a2_drop = a2 * mask2
        pred = a2_drop @ w3 + b3

        dp = 2.0 * (pred - yb) / batch_size
        gw3 = a2_drop.T @ dp
        gb3 = dp.sum(axis=0)
        da2 = (dp @ w3.T) * mask2
        dz2 = da2 * (z2 > 0.0)
        gw2 = a1_drop.T @ dz2
        gb2 = dz2.sum(axis=0)
        da1 = (dz2 @ w2.T) * mask1
        dz1 = da1 * (z1 > 0.0)
        gw1 = xb.T @ dz1
        gb1 = dz1.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2, gw3, gb3]

        lr = 0.003 if step < 2500 else 0.001
        for k, (param, grad) in enumerate(zip(params, grads)):
            adam_m[k] = beta1 * adam_m[k] + (1.0 - beta1) * grad
            adam_v[k] = beta2 * adam_v[k] + (1.0 - beta2) * (grad * grad)
            mhat = adam_m[k] / (1.0 - beta1**step)
            vhat = adam_v[k] / (1.0 - beta2**step)
            param -= lr * mhat / (np.sqrt(vhat) + eps)

    model = DropoutMLP(
        w1, b1, w2, b2, w3, b3, y_mean, y_std, dropout_rate
    )
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    def regression_metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float]:
        mse = float(np.mean((y - pred) ** 2))
        r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        return {f"{prefix}_mse": mse, f"{prefix}_r2": r2}

    metrics = regression_metrics(y_train, pred_train, "train_deterministic")
    metrics.update(regression_metrics(y_test, pred_test, "test_deterministic"))

    mc_rng = np.random.default_rng(seed + 1)
    mc_predictions = np.stack(
        [model.predict(x_test, model.sample_masks(len(x_test), mc_rng)) for _ in range(30)]
    )
    mc_mean = mc_predictions.mean(axis=0)
    metrics.update(regression_metrics(y_test, mc_mean, "test_mc_mean"))
    metrics["mean_mc_predictive_sd"] = float(
        mc_predictions.std(axis=0, ddof=1).mean()
    )
    metrics["median_mc_predictive_sd"] = float(
        np.median(mc_predictions.std(axis=0, ddof=1))
    )
    metrics["prediction_sd"] = float(pred_test.std(ddof=1))
    return model, metrics


def repeat_masks(
    masks: tuple[np.ndarray, np.ndarray], repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(masks[0], repeats, axis=0), np.repeat(masks[1], repeats, axis=0)


def evaluate(
    model: DropoutMLP,
    x: np.ndarray,
    *,
    stochastic: bool,
    rng: np.random.Generator,
    masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if not stochastic:
        return model.predict(x)
    if masks is None:
        masks = model.sample_masks(len(x), rng)
    return model.predict(x, masks)


def sobol_scores(
    model: DropoutMLP,
    *,
    n: int,
    seed: int,
    stochastic: bool,
    crn: bool,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    d = 10
    a = rng.random((n, d))
    b = rng.random((n, d))
    masks_a = model.sample_masks(n, rng) if stochastic else None
    fa = evaluate(model, a, stochastic=stochastic, rng=rng, masks=masks_a)
    fb = evaluate(model, b, stochastic=stochastic, rng=rng)
    variance = np.var(np.concatenate([fa, fb]), ddof=1)
    scores = np.empty(d)
    for j in range(d):
        ab = a.copy()
        ab[:, j] = b[:, j]
        masks_ab = masks_a if (stochastic and crn) else None
        fab = evaluate(model, ab, stochastic=stochastic, rng=rng, masks=masks_ab)
        scores[j] = np.mean((fa - fab) ** 2) / (2.0 * variance)
    return normalized(scores), n * (d + 2)


def gas_scores(
    model: DropoutMLP,
    *,
    m: int,
    q: int,
    seed: int,
    stochastic: bool,
    crn: bool,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = 10
    x = rng.random((m, d))
    nodes_raw, weights_raw = np.polynomial.legendre.leggauss(q)
    nodes = 0.5 * (nodes_raw + 1.0)
    weights = 0.5 * weights_raw
    masks_base = model.sample_masks(m, rng) if stochastic else None
    fz = evaluate(model, x, stochastic=stochastic, rng=rng, masks=masks_base)
    a_values = np.zeros((m, d))
    s_values = np.zeros((m, d))

    for j in range(d):
        left = np.repeat(x, q, axis=0)
        left[:, j] = (x[:, j, None] * nodes[None, :]).reshape(-1)
        right = np.repeat(x, q, axis=0)
        right[:, j] = (
            x[:, j, None] + (1.0 - x[:, j, None]) * nodes[None, :]
        ).reshape(-1)
        shared = repeat_masks(masks_base, q) if (stochastic and crn) else None
        f_left = evaluate(
            model, left, stochastic=stochastic, rng=rng, masks=shared
        ).reshape(m, q)
        f_right = evaluate(
            model, right, stochastic=stochastic, rng=rng, masks=shared
        ).reshape(m, q)
        left_values = left[:, j].reshape(m, q)
        right_values = right[:, j].reshape(m, q)
        q_left = (f_left - fz[:, None]) / (left_values - x[:, j, None])
        q_right = (f_right - fz[:, None]) / (right_values - x[:, j, None])
        a_values[:, j] = (
            np.sum(q_left * weights[None, :], axis=1) * x[:, j]
            + np.sum(q_right * weights[None, :], axis=1) * (1.0 - x[:, j])
        )
        s_values[:, j] = (
            np.sum(q_left**2 * weights[None, :], axis=1) * x[:, j]
            + np.sum(q_right**2 * weights[None, :], axis=1) * (1.0 - x[:, j])
        )

    activity = (a_values.T @ a_values) / m
    np.fill_diagonal(activity, s_values.mean(axis=0))
    activity = 0.5 * (activity + activity.T)
    eigenvalues, eigenvectors = np.linalg.eigh(activity)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = eigenvalues.sum()
    retained = (
        int(np.searchsorted(np.cumsum(eigenvalues) / total, 0.90) + 1)
        if total > 0
        else d
    )
    scores = np.sum(
        eigenvectors[:, :retained] ** 2 * eigenvalues[:retained], axis=1
    )
    return (
        normalized(scores),
        m * (1 + 2 * q * d),
        retained,
        eigenvalues,
        eigenvectors,
    )


def dgsm_scores(
    model: DropoutMLP,
    *,
    m: int,
    h: float,
    seed: int,
    stochastic: bool,
    crn: bool,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    x = rng.random((m, 10))
    values = np.empty(10)
    for j in range(10):
        plus = x.copy()
        minus = x.copy()
        plus[:, j] = np.minimum(1.0, x[:, j] + h)
        minus[:, j] = np.maximum(0.0, x[:, j] - h)
        shared = model.sample_masks(m, rng) if (stochastic and crn) else None
        fp = evaluate(model, plus, stochastic=stochastic, rng=rng, masks=shared)
        fm = evaluate(model, minus, stochastic=stochastic, rng=rng, masks=shared)
        derivative = (fp - fm) / (plus[:, j] - minus[:, j])
        values[j] = np.mean(derivative**2)
    return normalized(values), 2 * m * 10


def subspace_metrics(
    reference_vectors: np.ndarray,
    estimated_vectors: np.ndarray,
    dimension: int,
) -> tuple[float, float, float]:
    ref = reference_vectors[:, :dimension]
    est = estimated_vectors[:, :dimension]
    singular_values = np.linalg.svd(ref.T @ est, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    overlap = float(np.mean(singular_values**2))
    max_angle = float(np.degrees(np.arccos(singular_values.min())))
    top_direction_cosine = float(abs(reference_vectors[:, 0] @ estimated_vectors[:, 0]))
    return overlap, max_angle, top_direction_cosine


def paired_contrasts(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    comparisons = [
        ("mc_dropout_crn", "deterministic", "crn_minus_deterministic"),
        ("mc_dropout_independent", "mc_dropout_crn", "independent_minus_crn"),
    ]
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    for key, subset in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_columns, keys))
        for metric in metrics:
            pivot = subset.pivot(index="repeat", columns="condition", values=metric)
            for minuend, subtrahend, label in comparisons:
                delta = (pivot[minuend] - pivot[subtrahend]).dropna()
                mean = float(delta.mean())
                sd = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
                half_width = 1.96 * sd / math.sqrt(len(delta)) if len(delta) > 1 else 0.0
                rows.append(
                    {
                        **group_values,
                        "metric": metric,
                        "contrast": label,
                        "n": len(delta),
                        "mean_difference": mean,
                        "sd_difference": sd,
                        "normal_95ci_low": mean - half_width,
                        "normal_95ci_high": mean + half_width,
                    }
                )
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, gas_summary: pd.DataFrame, out: Path) -> None:
    width, height = 1800, 920
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy = "#17365D"
    blue = "#4F81BD"
    orange = "#F28E2B"
    green = "#59A14F"
    gray = "#666666"
    draw.text((70, 35), "GAS under Monte Carlo dropout", fill=navy, font=font(34, True))

    conditions = ["deterministic", "mc_dropout_crn", "mc_dropout_independent"]
    labels = ["Deterministic", "MC dropout\npaired masks", "MC dropout\nindependent masks"]
    methods = ["Sobol", "GAS", "DGSM"]
    colors = {"Sobol": blue, "GAS": orange, "DGSM": green}

    x0, y0, w, h = 100, 155, 1000, 590
    draw.text((x0, 105), "Feature-ranking stability", fill=navy, font=font(21, True))
    draw.line((x0, y0 + h, x0 + w, y0 + h), fill="#333333", width=2)
    draw.line((x0, y0, x0, y0 + h), fill="#333333", width=2)
    group_w = w / len(conditions)
    bar_w = group_w / 4.4
    for i, (condition, label) in enumerate(zip(conditions, labels)):
        gx = x0 + i * group_w + 35
        for k, method in enumerate(methods):
            row = summary[
                (summary.condition == condition) & (summary.method == method)
            ].iloc[0]
            value = max(0.0, float(row.kendall_tau_mean))
            bh = value * h
            bx = gx + k * bar_w
            draw.rectangle((bx, y0 + h - bh, bx + bar_w - 6, y0 + h), fill=colors[method])
            draw.text((bx + 4, y0 + h - bh - 28), f"{value:.2f}", fill=gray, font=font(15))
        for line_no, line in enumerate(label.split("\n")):
            draw.text((gx, y0 + h + 14 + line_no * 22), line, fill="#222222", font=font(17))
    for k, method in enumerate(methods):
        lx = x0 + 210 + k * 210
        draw.rectangle((lx, 830, lx + 25, 855), fill=colors[method])
        draw.text((lx + 35, 828), method, fill="#222222", font=font(18))

    rx, ry = 1190, 160
    draw.text((rx, 105), "GAS direction stability", fill=navy, font=font(21, True))
    for i, condition in enumerate(conditions):
        row = gas_summary[gas_summary.condition == condition].iloc[0]
        label = labels[i].replace("\n", " ")
        yy = ry + i * 180
        draw.text((rx, yy), label, fill="#222222", font=font(19, True))
        draw.text(
            (rx, yy + 42),
            f"Subspace overlap: {row.subspace_overlap_mean:.3f}",
            fill="#222222",
            font=font(18),
        )
        draw.text(
            (rx, yy + 80),
            f"Max principal angle: {row.max_principal_angle_mean:.1f} deg",
            fill="#222222",
            font=font(18),
        )
        draw.text(
            (rx, yy + 118),
            f"Top-direction |cos|: {row.top_direction_cosine_mean:.3f}",
            fill="#222222",
            font=font(18),
        )
    image.save(out)


def run(
    out_dir: Path,
    *,
    repeats: int = 12,
    steps: int = 3500,
    dropout_rate: float = 0.10,
    training_seed: int = 20260930,
) -> None:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, model_metrics = train_dropout_mlp(
        seed=training_seed, steps=steps, dropout_rate=dropout_rate
    )

    sobol_reference, sobol_reference_evals = sobol_scores(
        model, n=32768, seed=911, stochastic=False, crn=True
    )
    (
        gas_reference,
        gas_reference_evals,
        gas_reference_retained,
        gas_reference_eigenvalues,
        gas_reference_vectors,
    ) = gas_scores(model, m=4096, q=8, seed=913, stochastic=False, crn=True)
    dgsm_reference, dgsm_reference_evals = dgsm_scores(
        model, m=32768, h=0.005, seed=914, stochastic=False, crn=True
    )
    method_references = {
        "Sobol": sobol_reference,
        "GAS": gas_reference,
        "DGSM": dgsm_reference,
    }

    conditions = [
        ("deterministic", False, True),
        ("mc_dropout_crn", True, True),
        ("mc_dropout_independent", True, False),
    ]
    score_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    gas_rows: list[dict[str, object]] = []
    eig_rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        for condition_index, (condition, stochastic, crn) in enumerate(conditions):
            seed = 20000 + 100 * repeat + 10 * condition_index
            sobol, sobol_evals = sobol_scores(
                model, n=512, seed=seed + 1, stochastic=stochastic, crn=crn
            )
            (
                gas,
                gas_evals,
                retained,
                eigenvalues,
                eigenvectors,
            ) = gas_scores(
                model, m=64, q=4, seed=seed + 2, stochastic=stochastic, crn=crn
            )
            dgsm, dgsm_evals = dgsm_scores(
                model, m=256, h=0.02, seed=seed + 3, stochastic=stochastic, crn=crn
            )

            for method, scores, eval_count in (
                ("Sobol", sobol, sobol_evals),
                ("GAS", gas, gas_evals),
                ("DGSM", dgsm, dgsm_evals),
            ):
                reference = method_references[method]
                metrics = {
                    "repeat": repeat,
                    "condition": condition,
                    "method": method,
                    "kendall_tau": kendall_tau(scores, reference),
                    "kendall_tau_relevant": kendall_tau(scores[:5], reference[:5]),
                    "l1_error": float(np.sum(np.abs(scores - reference))),
                    "top5_recovery": top_k_recovery(scores),
                    "eval_count": eval_count,
                }
                metric_rows.append(metrics)
                for feature_index, score in enumerate(scores):
                    score_rows.append(
                        {
                            "repeat": repeat,
                            "condition": condition,
                            "method": method,
                            "feature": FEATURES[feature_index],
                            "feature_index": feature_index,
                            "score": float(score),
                        }
                    )

            overlap, angle, cosine = subspace_metrics(
                gas_reference_vectors, eigenvectors, gas_reference_retained
            )
            gas_rows.append(
                {
                    "repeat": repeat,
                    "condition": condition,
                    "retained_90": retained,
                    "subspace_dimension": gas_reference_retained,
                    "subspace_overlap": overlap,
                    "max_principal_angle": angle,
                    "top_direction_cosine": cosine,
                }
            )
            for eigen_index, value in enumerate(eigenvalues, start=1):
                eig_rows.append(
                    {
                        "repeat": repeat,
                        "condition": condition,
                        "eigen_index": eigen_index,
                        "eigenvalue": float(value),
                        "retained_90": retained,
                    }
                )

    scores_df = pd.DataFrame(score_rows)
    metrics_df = pd.DataFrame(metric_rows)
    gas_df = pd.DataFrame(gas_rows)
    eig_df = pd.DataFrame(eig_rows)
    summary = (
        metrics_df.groupby(["condition", "method"], as_index=False)
        .agg(
            kendall_tau_mean=("kendall_tau", "mean"),
            kendall_tau_sd=("kendall_tau", "std"),
            kendall_tau_relevant_mean=("kendall_tau_relevant", "mean"),
            kendall_tau_relevant_sd=("kendall_tau_relevant", "std"),
            l1_mean=("l1_error", "mean"),
            l1_sd=("l1_error", "std"),
            top5_recovery_mean=("top5_recovery", "mean"),
            eval_count=("eval_count", "first"),
        )
    )
    gas_summary = (
        gas_df.groupby("condition", as_index=False)
        .agg(
            retained_90_mean=("retained_90", "mean"),
            retained_90_sd=("retained_90", "std"),
            subspace_overlap_mean=("subspace_overlap", "mean"),
            subspace_overlap_sd=("subspace_overlap", "std"),
            max_principal_angle_mean=("max_principal_angle", "mean"),
            max_principal_angle_sd=("max_principal_angle", "std"),
            top_direction_cosine_mean=("top_direction_cosine", "mean"),
            top_direction_cosine_sd=("top_direction_cosine", "std"),
        )
    )
    method_contrasts = paired_contrasts(
        metrics_df,
        group_columns=["method"],
        metrics=["kendall_tau", "l1_error", "top5_recovery"],
    )
    gas_contrasts = paired_contrasts(
        gas_df,
        group_columns=[],
        metrics=[
            "retained_90",
            "subspace_overlap",
            "max_principal_angle",
            "top_direction_cosine",
        ],
    )
    mean_scores = (
        scores_df.groupby(
            ["condition", "method", "feature", "feature_index"], as_index=False
        )
        .agg(score_mean=("score", "mean"), score_sd=("score", "std"))
    )
    reference_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_index": np.arange(10),
            "sobol_reference": sobol_reference,
            "gas_reference": gas_reference,
            "dgsm_reference": dgsm_reference,
        }
    )

    scores_df.to_csv(out_dir / "method_scores.csv", index=False)
    metrics_df.to_csv(out_dir / "method_stability.csv", index=False)
    summary.to_csv(out_dir / "stability_summary.csv", index=False)
    mean_scores.to_csv(out_dir / "mean_scores.csv", index=False)
    gas_df.to_csv(out_dir / "gas_subspace_stability.csv", index=False)
    gas_summary.to_csv(out_dir / "gas_subspace_summary.csv", index=False)
    method_contrasts.to_csv(out_dir / "protocol_contrasts.csv", index=False)
    gas_contrasts.to_csv(out_dir / "gas_subspace_contrasts.csv", index=False)
    eig_df.to_csv(out_dir / "gas_eigenvalues.csv", index=False)
    reference_df.to_csv(out_dir / "reference_scores.csv", index=False)
    np.savez_compressed(
        out_dir / "gas_reference_subspace.npz",
        eigenvalues=gas_reference_eigenvalues,
        eigenvectors=gas_reference_vectors,
    )

    elapsed = time.perf_counter() - started
    model_metrics.update(
        {
            "training_seed": training_seed,
            "training_steps": steps,
            "dropout_rate": dropout_rate,
            "mc_test_passes": 30,
            "repeats": repeats,
            "reference_sobol_evaluations": sobol_reference_evals,
            "reference_gas_evaluations": gas_reference_evals,
            "reference_gas_retained_90": gas_reference_retained,
            "reference_dgsm_evaluations": dgsm_reference_evals,
            "sobol_budget": 512 * 12,
            "gas_budget": 64 * (1 + 2 * 4 * 10),
            "dgsm_budget": 2 * 256 * 10,
            "runtime_seconds": elapsed,
            "stochasticity": "Monte Carlo dropout masks in both hidden layers",
        }
    )
    (out_dir / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2), encoding="utf-8"
    )
    make_plot(summary, gas_summary, out_dir / "mc_dropout_stability.png")

    print(json.dumps(model_metrics, indent=2))
    print("\nStability summary:")
    print(summary.to_string(index=False))
    print("\nGAS subspace summary:")
    print(gas_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--dropout-rate", type=float, default=0.10)
    parser.add_argument("--training-seed", type=int, default=20260930)
    args = parser.parse_args()
    run(
        args.out,
        repeats=args.repeats,
        steps=args.steps,
        dropout_rate=args.dropout_rate,
        training_seed=args.training_seed,
    )
