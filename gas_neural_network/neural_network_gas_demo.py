"""Minimal GAS experiment on a trained ReLU neural network.

The data-generating function is the Friedman-1 benchmark with 10 independent
uniform inputs; only x1--x5 are relevant.  A small MLP is trained in NumPy and
then treated as a black-box model.  GAS, Sobol total-effect, and DGSM are
compared at approximately equal model-evaluation budgets under deterministic
evaluation and additive output noise, with either common random numbers (CRN)
or independent noise in paired evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUT = Path(__file__).resolve().parent / "results"
FEATURES = tuple(f"x{i}" for i in range(1, 11))
RELEVANT = set(range(5))


def friedman_function(x: np.ndarray) -> np.ndarray:
    return (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
    )


@dataclass
class MLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    y_mean: float
    y_std: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=float) - 0.5) / math.sqrt(1.0 / 12.0)
        a1 = np.maximum(0.0, xs @ self.w1 + self.b1)
        a2 = np.maximum(0.0, a1 @ self.w2 + self.b2)
        standardized = (a2 @ self.w3 + self.b3).ravel()
        return standardized * self.y_std + self.y_mean


def train_mlp(seed: int = 20260916, steps: int = 3500) -> tuple[MLP, dict[str, float]]:
    rng = np.random.default_rng(seed)
    x_train = rng.random((6000, 10))
    x_test = rng.random((2500, 10))
    y_train = friedman_function(x_train)
    y_test = friedman_function(x_test)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    target = (y_train - y_mean) / y_std

    h1 = h2 = 32
    w1 = rng.normal(0.0, math.sqrt(2.0 / 10), (10, h1))
    b1 = np.zeros(h1)
    w2 = rng.normal(0.0, math.sqrt(2.0 / h1), (h1, h2))
    b2 = np.zeros(h2)
    w3 = rng.normal(0.0, math.sqrt(2.0 / h2), (h2, 1))
    b3 = np.zeros(1)
    params = [w1, b1, w2, b2, w3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    batch_size = 256

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(x_train), size=batch_size)
        xb = (x_train[idx] - 0.5) / math.sqrt(1.0 / 12.0)
        yb = target[idx, None]

        z1 = xb @ w1 + b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ w2 + b2
        a2 = np.maximum(0.0, z2)
        pred = a2 @ w3 + b3
        dp = 2.0 * (pred - yb) / batch_size
        gw3 = a2.T @ dp
        gb3 = dp.sum(axis=0)
        da2 = dp @ w3.T
        dz2 = da2 * (z2 > 0.0)
        gw2 = a1.T @ dz2
        gb2 = dz2.sum(axis=0)
        da1 = dz2 @ w2.T
        dz1 = da1 * (z1 > 0.0)
        gw1 = xb.T @ dz1
        gb1 = dz1.sum(axis=0)
        grads = [gw1, gb1, gw2, gb2, gw3, gb3]

        lr = 0.003 if step < 2500 else 0.001
        for k, (p, g) in enumerate(zip(params, grads)):
            m[k] = beta1 * m[k] + (1.0 - beta1) * g
            v[k] = beta2 * v[k] + (1.0 - beta2) * (g * g)
            mhat = m[k] / (1.0 - beta1**step)
            vhat = v[k] / (1.0 - beta2**step)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    model = MLP(w1, b1, w2, b2, w3, b3, y_mean, y_std)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    def metrics(y: np.ndarray, p: np.ndarray, prefix: str) -> dict[str, float]:
        mse = float(np.mean((y - p) ** 2))
        r2 = 1.0 - float(np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
        return {f"{prefix}_mse": mse, f"{prefix}_r2": r2}

    out = metrics(y_train, pred_train, "train")
    out.update(metrics(y_test, pred_test, "test"))
    out["prediction_std"] = float(pred_test.std(ddof=1))
    return model, out


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = values.sum()
    return values / total if total > 0 else np.full_like(values, 1.0 / len(values))


def noisy_prediction(
    model: MLP,
    x: np.ndarray,
    *,
    noise_sd: float,
    rng: np.random.Generator,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    pred = model.predict(x)
    if noise_sd == 0.0:
        return pred
    if noise is None:
        noise = rng.standard_normal(len(x))
    return pred + noise_sd * noise


def sobol_scores(
    model: MLP,
    *,
    n: int,
    seed: int,
    noise_sd: float = 0.0,
    crn: bool = True,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    d = 10
    a = rng.random((n, d))
    b = rng.random((n, d))
    eps_a = rng.standard_normal(n) if noise_sd else None
    fa = noisy_prediction(model, a, noise_sd=noise_sd, rng=rng, noise=eps_a)
    fb = noisy_prediction(model, b, noise_sd=noise_sd, rng=rng)
    variance = np.var(np.concatenate([fa, fb]), ddof=1)
    scores = np.empty(d)
    for j in range(d):
        ab = a.copy()
        ab[:, j] = b[:, j]
        eps_ab = eps_a if crn else None
        fab = noisy_prediction(model, ab, noise_sd=noise_sd, rng=rng, noise=eps_ab)
        scores[j] = np.mean((fa - fab) ** 2) / (2.0 * variance)
    return normalized(scores), n * (d + 2)


def gas_scores(
    model: MLP,
    *,
    m: int,
    q: int,
    seed: int,
    noise_sd: float = 0.0,
    crn: bool = True,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = 10
    x = rng.random((m, d))
    nodes_raw, weights_raw = np.polynomial.legendre.leggauss(q)
    nodes = 0.5 * (nodes_raw + 1.0)
    weights = 0.5 * weights_raw
    eps_base = rng.standard_normal(m) if noise_sd else None
    fz = noisy_prediction(model, x, noise_sd=noise_sd, rng=rng, noise=eps_base)
    a_values = np.zeros((m, d))
    s_values = np.zeros((m, d))

    for j in range(d):
        left = np.repeat(x, q, axis=0)
        left[:, j] = (x[:, j, None] * nodes[None, :]).reshape(-1)
        right = np.repeat(x, q, axis=0)
        right[:, j] = (
            x[:, j, None] + (1.0 - x[:, j, None]) * nodes[None, :]
        ).reshape(-1)
        shared = np.repeat(eps_base, q) if (noise_sd and crn) else None
        f_left = noisy_prediction(model, left, noise_sd=noise_sd, rng=rng, noise=shared).reshape(m, q)
        f_right = noisy_prediction(model, right, noise_sd=noise_sd, rng=rng, noise=shared).reshape(m, q)
        left_values = left[:, j].reshape(m, q)
        right_values = right[:, j].reshape(m, q)
        q_left = (f_left - fz[:, None]) / (left_values - x[:, j, None])
        q_right = (f_right - fz[:, None]) / (right_values - x[:, j, None])
        left_jac = x[:, j]
        right_jac = 1.0 - x[:, j]
        a_values[:, j] = (
            np.sum(q_left * weights[None, :], axis=1) * left_jac
            + np.sum(q_right * weights[None, :], axis=1) * right_jac
        )
        s_values[:, j] = (
            np.sum(q_left**2 * weights[None, :], axis=1) * left_jac
            + np.sum(q_right**2 * weights[None, :], axis=1) * right_jac
        )

    c = (a_values.T @ a_values) / m
    np.fill_diagonal(c, s_values.mean(axis=0))
    c = 0.5 * (c + c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(c)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    retained = int(np.searchsorted(cumulative, 0.90) + 1)
    scores = np.sum(
        eigenvectors[:, :retained] ** 2 * eigenvalues[:retained], axis=1
    )
    return normalized(scores), m * (1 + 2 * q * d), retained, eigenvalues


def dgsm_scores(
    model: MLP,
    *,
    m: int,
    h: float,
    seed: int,
    noise_sd: float = 0.0,
    crn: bool = True,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    x = rng.random((m, 10))
    eps_base = rng.standard_normal(m) if noise_sd else None
    values = np.empty(10)
    for j in range(10):
        plus = x.copy()
        minus = x.copy()
        plus[:, j] = np.minimum(1.0, x[:, j] + h)
        minus[:, j] = np.maximum(0.0, x[:, j] - h)
        shared = eps_base if (noise_sd and crn) else None
        fp = noisy_prediction(model, plus, noise_sd=noise_sd, rng=rng, noise=shared)
        fm = noisy_prediction(model, minus, noise_sd=noise_sd, rng=rng, noise=shared)
        derivative = (fp - fm) / (plus[:, j] - minus[:, j])
        values[j] = np.mean(derivative**2)
    return normalized(values), 2 * m * 10


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            product = (a[i] - a[j]) * (b[i] - b[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else 1.0


def top_k_recovery(scores: np.ndarray, k: int = 5) -> float:
    selected = set(np.argsort(scores)[::-1][:k].tolist())
    return len(selected & RELEVANT) / len(RELEVANT)


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_plot(reference: np.ndarray, deterministic: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    width, height = 1800, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(34, True)
    label_font = font(21, False)
    small_font = font(17, False)
    bold_font = font(18, True)
    navy = "#17365D"
    blue = "#4F81BD"
    orange = "#F28E2B"
    green = "#59A14F"
    gray = "#777777"
    draw.text((70, 35), "GAS on a trained ReLU neural network", fill=navy, font=title_font)

    # Left: deterministic score comparison.
    x0, y0, w, h = 80, 130, 780, 600
    draw.text((x0, 90), "Deterministic normalized sensitivity scores", fill=navy, font=bold_font)
    draw.line((x0, y0 + h, x0 + w, y0 + h), fill="#333333", width=2)
    draw.line((x0, y0, x0, y0 + h), fill="#333333", width=2)
    methods = ["Reference", "Sobol", "GAS", "DGSM"]
    colors = ["#222222", blue, orange, green]
    series = {"Reference": reference}
    for method in methods[1:]:
        subset = deterministic[deterministic.method == method].sort_values("feature_index")
        series[method] = subset.score.to_numpy()
    ymax = max(max(v) for v in series.values()) * 1.12
    group_w = w / 10
    bar_w = group_w / 5
    for i, name in enumerate(FEATURES):
        gx = x0 + i * group_w + 8
        for k, method in enumerate(methods):
            value = float(series[method][i])
            bh = h * value / ymax
            bx = gx + k * bar_w
            draw.rectangle((bx, y0 + h - bh, bx + bar_w - 2, y0 + h), fill=colors[k])
        draw.text((gx + 8, y0 + h + 12), name, fill="#222222", font=small_font)
    for k, method in enumerate(methods):
        lx = x0 + 70 + k * 170
        draw.rectangle((lx, y0 + h + 65, lx + 22, y0 + h + 87), fill=colors[k])
        draw.text((lx + 30, y0 + h + 63), method, fill="#222222", font=small_font)

    # Right: rank stability by condition.
    rx0, ry0, rw, rh = 970, 130, 740, 600
    draw.text((rx0, 90), "Mean Kendall tau vs each method's reference", fill=navy, font=bold_font)
    draw.line((rx0, ry0 + rh, rx0 + rw, ry0 + rh), fill="#333333", width=2)
    draw.line((rx0, ry0, rx0, ry0 + rh), fill="#333333", width=2)
    conditions = [
        "Deterministic",
        "5% noise\nCRN",
        "5% noise\nIndependent",
        "15% noise\nCRN",
        "15% noise\nIndependent",
    ]
    keys = [
        (0.0, "deterministic"),
        (0.05, "CRN"),
        (0.05, "independent"),
        (0.15, "CRN"),
        (0.15, "independent"),
    ]
    group_w = rw / len(keys)
    bar_w = group_w / 4
    method_colors = {"Sobol": blue, "GAS": orange, "DGSM": green}
    for i, ((level, protocol), label) in enumerate(zip(keys, conditions)):
        gx = rx0 + i * group_w + 12
        for k, method in enumerate(("Sobol", "GAS", "DGSM")):
            row = summary[
                (summary.noise_fraction == level)
                & (summary.protocol == protocol)
                & (summary.method == method)
            ].iloc[0]
            value = float(row.kendall_tau_mean)
            bh = rh * max(value, 0.0)
            bx = gx + k * bar_w
            draw.rectangle((bx, ry0 + rh - bh, bx + bar_w - 3, ry0 + rh), fill=method_colors[method])
        lines = label.split("\n")
        for line_no, line in enumerate(lines):
            draw.text((gx, ry0 + rh + 12 + line_no * 19), line, fill="#222222", font=small_font)
    for k, method in enumerate(("Sobol", "GAS", "DGSM")):
        lx = rx0 + 130 + k * 170
        draw.rectangle((lx, ry0 + rh + 70, lx + 22, ry0 + rh + 92), fill=method_colors[method])
        draw.text((lx + 30, ry0 + rh + 68), method, fill="#222222", font=small_font)
    draw.text((80, 835), "Relevant features are x1-x5; x6-x10 are null features. Error bars are reported in CSV summaries.", fill=gray, font=label_font)
    image.save(out)


def run(out_dir: Path, repeats: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, model_metrics = train_mlp()
    prediction_sd = model_metrics["prediction_std"]

    sobol_reference, sobol_reference_evals = sobol_scores(
        model, n=32768, seed=811, noise_sd=0.0
    )
    gas_reference, gas_reference_evals, gas_reference_retained, _ = gas_scores(
        model, m=4096, q=8, seed=813, noise_sd=0.0
    )
    dgsm_reference, dgsm_reference_evals = dgsm_scores(
        model, m=32768, h=0.005, seed=814, noise_sd=0.0
    )
    method_references = {
        "Sobol": sobol_reference,
        "GAS": gas_reference,
        "DGSM": dgsm_reference,
    }
    true_reference_model = MLP(
        np.empty((0, 0)), np.empty(0), np.empty((0, 0)), np.empty(0),
        np.empty((0, 0)), np.empty(0), 0.0, 1.0,
    )
    true_reference_model.predict = friedman_function  # type: ignore[method-assign]
    true_reference, _ = sobol_scores(true_reference_model, n=32768, seed=812, noise_sd=0.0)

    conditions = [
        (0.0, "deterministic", True),
        (0.05, "CRN", True),
        (0.05, "independent", False),
        (0.15, "CRN", True),
        (0.15, "independent", False),
    ]
    rows: list[dict[str, object]] = []
    eig_rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        for noise_fraction, protocol, crn in conditions:
            noise_sd = noise_fraction * prediction_sd
            seed = 10000 + repeat * 100 + int(noise_fraction * 1000) + (0 if crn else 33)
            sobol, sobol_evals = sobol_scores(
                model, n=512, seed=seed + 1, noise_sd=noise_sd, crn=crn
            )
            gas, gas_evals, retained, eigenvalues = gas_scores(
                model, m=64, q=4, seed=seed + 2, noise_sd=noise_sd, crn=crn
            )
            dgsm, dgsm_evals = dgsm_scores(
                model, m=256, h=0.02, seed=seed + 3, noise_sd=noise_sd, crn=crn
            )
            for method, scores, eval_count in (
                ("Sobol", sobol, sobol_evals),
                ("GAS", gas, gas_evals),
                ("DGSM", dgsm, dgsm_evals),
            ):
                method_reference = method_references[method]
                tau = kendall_tau(scores, method_reference)
                tau_relevant = kendall_tau(scores[:5], method_reference[:5])
                l1 = float(np.sum(np.abs(scores - method_reference)))
                recovery = top_k_recovery(scores)
                for j, score in enumerate(scores):
                    rows.append(
                        {
                            "repeat": repeat,
                            "noise_fraction": noise_fraction,
                            "protocol": protocol,
                            "method": method,
                            "feature": FEATURES[j],
                            "feature_index": j,
                            "score": float(score),
                            "kendall_tau": tau,
                            "kendall_tau_relevant": tau_relevant,
                            "l1_error": l1,
                            "top5_recovery": recovery,
                            "eval_count": eval_count,
                        }
                    )
            for j, value in enumerate(eigenvalues):
                eig_rows.append(
                    {
                        "repeat": repeat,
                        "noise_fraction": noise_fraction,
                        "protocol": protocol,
                        "eigen_index": j + 1,
                        "eigenvalue": float(value),
                        "retained_90": retained,
                    }
                )

    scores_df = pd.DataFrame(rows)
    eig_df = pd.DataFrame(eig_rows)
    metric_df = scores_df.drop_duplicates(
        ["repeat", "noise_fraction", "protocol", "method"]
    )
    summary = (
        metric_df.groupby(["noise_fraction", "protocol", "method"], as_index=False)
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
    mean_scores = (
        scores_df.groupby(
            ["noise_fraction", "protocol", "method", "feature", "feature_index"],
            as_index=False,
        )
        .agg(score_mean=("score", "mean"), score_sd=("score", "std"))
    )
    deterministic = mean_scores[
        (mean_scores.noise_fraction == 0.0)
        & (mean_scores.protocol == "deterministic")
    ].rename(columns={"score_mean": "score"})

    reference_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_index": np.arange(10),
            "trained_mlp_sobol_reference": sobol_reference,
            "trained_mlp_gas_reference": gas_reference,
            "trained_mlp_dgsm_reference": dgsm_reference,
            "true_function_sobol_reference": true_reference,
        }
    )
    scores_df.to_csv(out_dir / "method_scores.csv", index=False)
    summary.to_csv(out_dir / "stability_summary.csv", index=False)
    mean_scores.to_csv(out_dir / "mean_scores.csv", index=False)
    eig_df.to_csv(out_dir / "gas_eigenvalues.csv", index=False)
    reference_df.to_csv(out_dir / "reference_scores.csv", index=False)
    model_metrics.update(
        {
            "training_seed": 20260916,
            "training_steps": 3500,
            "reference_sobol_evaluations": sobol_reference_evals,
            "reference_gas_evaluations": gas_reference_evals,
            "reference_gas_retained_90": gas_reference_retained,
            "reference_dgsm_evaluations": dgsm_reference_evals,
            "repeats": repeats,
            "sobol_budget": 512 * 12,
            "gas_budget": 64 * (1 + 2 * 4 * 10),
            "dgsm_budget": 2 * 256 * 10,
            "noise_definition": "Gaussian additive output noise as a fraction of test prediction SD",
        }
    )
    (out_dir / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2), encoding="utf-8"
    )
    make_plot(sobol_reference, deterministic, summary, out_dir / "gas_neural_network_demo.png")

    print(json.dumps(model_metrics, indent=2))
    print("\nReference scores:")
    print(reference_df.to_string(index=False))
    print("\nStability summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    run(args.out, repeats=args.repeats)
