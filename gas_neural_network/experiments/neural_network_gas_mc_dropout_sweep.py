"""Multi-seed, multi-dropout robustness sweep for the stage-2 GAS study.

The statistical unit in the aggregate tables is the independently trained
network, not each sensitivity repeat.  Each network first averages over its 12
sensitivity replicates; uncertainty across models is then summarized with a
Student-t interval.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from neural_network_gas_demo import font
from neural_network_gas_mc_dropout import run as run_single


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "dropout_sweep"
DEFAULT_SEEDS = [2026101401, 2026101402, 2026101403, 2026101404, 2026101405]
DEFAULT_RATES = [0.05, 0.10, 0.20, 0.30]

# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_critical_95(n: int) -> float:
    if n <= 1:
        return 0.0
    return T95.get(n - 1, 1.96)


def aggregate_models(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, subset in frame.groupby(groups, dropna=False, sort=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, object] = dict(zip(groups, keys))
        n_models = int(subset.training_seed.nunique())
        row["n_models"] = n_models
        critical = t_critical_95(n_models)
        for metric in metrics:
            values = subset[metric].astype(float).to_numpy()
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half_width = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_across_models_mean"] = mean
            row[f"{metric}_across_models_sd"] = sd
            row[f"{metric}_t95_low"] = mean - half_width
            row[f"{metric}_t95_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    rates: list[float],
    series: dict[str, tuple[list[float], list[float], str]],
    y_min: float,
    y_max: float,
    y_label: str,
) -> None:
    navy = "#17365D"
    gray = "#666666"
    draw.text((x0, y0 - 55), title, fill=navy, font=font(20, True))
    draw.line((x0, y0, x0, y0 + height), fill="#333333", width=2)
    draw.line((x0, y0 + height, x0 + width, y0 + height), fill="#333333", width=2)
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        yy = y0 + height - int(height * tick / 5)
        draw.line((x0 - 6, yy, x0, yy), fill="#333333", width=2)
        draw.text((x0 - 72, yy - 10), f"{value:.1f}", fill=gray, font=font(14))
    draw.text((x0 - 80, y0 - 30), y_label, fill=gray, font=font(14))

    x_positions = [x0 + int(width * (i + 0.5) / len(rates)) for i in range(len(rates))]
    for xx, rate in zip(x_positions, rates):
        draw.line((xx, y0 + height, xx, y0 + height + 6), fill="#333333", width=2)
        draw.text((xx - 22, y0 + height + 12), f"{int(rate * 100)}%", fill="#222222", font=font(16))
    draw.text((x0 + width // 2 - 45, y0 + height + 48), "dropout rate", fill=gray, font=font(15))

    for label, (means, sds, color) in series.items():
        points: list[tuple[int, int]] = []
        for xx, mean, sd in zip(x_positions, means, sds):
            scaled = (mean - y_min) / (y_max - y_min)
            yy = y0 + height - int(height * np.clip(scaled, 0.0, 1.0))
            points.append((xx, yy))
            err = int(height * sd / (y_max - y_min))
            draw.line((xx, yy - err, xx, yy + err), fill=color, width=2)
            draw.line((xx - 6, yy - err, xx + 6, yy - err), fill=color, width=2)
            draw.line((xx - 6, yy + err, xx + 6, yy + err), fill=color, width=2)
            draw.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill=color)
        if len(points) > 1:
            draw.line(points, fill=color, width=4)


def make_plot(method_summary: pd.DataFrame, gas_summary: pd.DataFrame, out: Path) -> None:
    rates = sorted(method_summary.dropout_rate.unique().tolist())
    conditions = [
        ("deterministic", "Deterministic", "#555555"),
        ("mc_dropout_crn", "Paired masks", "#4F81BD"),
        ("mc_dropout_independent", "Independent masks", "#F28E2B"),
    ]

    def method_series(metric: str) -> dict[str, tuple[list[float], list[float], str]]:
        answer: dict[str, tuple[list[float], list[float], str]] = {}
        for condition, label, color in conditions:
            subset = method_summary[
                (method_summary.method == "GAS")
                & (method_summary.condition == condition)
            ].set_index("dropout_rate")
            means = [float(subset.loc[rate, f"{metric}_across_models_mean"]) for rate in rates]
            sds = [float(subset.loc[rate, f"{metric}_across_models_sd"]) for rate in rates]
            answer[label] = (means, sds, color)
        return answer

    def gas_series(metric: str) -> dict[str, tuple[list[float], list[float], str]]:
        answer: dict[str, tuple[list[float], list[float], str]] = {}
        for condition, label, color in conditions:
            subset = gas_summary[gas_summary.condition == condition].set_index("dropout_rate")
            means = [float(subset.loc[rate, f"{metric}_across_models_mean"]) for rate in rates]
            sds = [float(subset.loc[rate, f"{metric}_across_models_sd"]) for rate in rates]
            answer[label] = (means, sds, color)
        return answer

    image = Image.new("RGB", (1900, 980), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "GAS robustness across trained dropout networks", fill="#17365D", font=font(34, True))
    draw.text(
        (72, 85),
        "Five independently trained models per dropout rate; error bars show between-model SD.",
        fill="#666666",
        font=font(18),
    )
    draw_panel(
        draw,
        x0=100,
        y0=205,
        width=470,
        height=580,
        title="Feature-ranking stability",
        rates=rates,
        series=method_series("kendall_tau_mean"),
        y_min=0.0,
        y_max=1.0,
        y_label="Kendall tau",
    )
    draw_panel(
        draw,
        x0=720,
        y0=205,
        width=470,
        height=580,
        title="GAS subspace recovery",
        rates=rates,
        series=gas_series("subspace_overlap_mean"),
        y_min=0.0,
        y_max=1.0,
        y_label="overlap",
    )
    draw_panel(
        draw,
        x0=1340,
        y0=205,
        width=470,
        height=580,
        title="Estimated activity dimension",
        rates=rates,
        series=gas_series("retained_90_mean"),
        y_min=0.0,
        y_max=10.0,
        y_label="dimensions",
    )
    legend_x = 565
    for index, (_, label, color) in enumerate(conditions):
        xx = legend_x + index * 270
        draw.line((xx, 900, xx + 50, 900), fill=color, width=5)
        draw.ellipse((xx + 19, 894, xx + 31, 906), fill=color)
        draw.text((xx + 62, 888), label, fill="#222222", font=font(17))
    image.save(out)


def run_sweep(
    out_dir: Path,
    *,
    seeds: list[int],
    rates: list[float],
    repeats: int,
    steps: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    method_frames: list[pd.DataFrame] = []
    gas_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    gas_contrast_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []

    total = len(seeds) * len(rates)
    completed = 0
    for rate in rates:
        rate_label = f"p{int(round(rate * 100)):02d}"
        for seed in seeds:
            completed += 1
            config_dir = out_dir / "runs" / rate_label / f"seed_{seed}"
            config_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{completed}/{total}] dropout={rate:.2f}, training_seed={seed}", flush=True)
            with (config_dir / "run.log").open("w", encoding="utf-8") as log_file:
                with contextlib.redirect_stdout(log_file):
                    run_single(
                        config_dir,
                        repeats=repeats,
                        steps=steps,
                        dropout_rate=rate,
                        training_seed=seed,
                    )

            method = pd.read_csv(config_dir / "stability_summary.csv")
            method.insert(0, "training_seed", seed)
            method.insert(0, "dropout_rate", rate)
            method_frames.append(method)

            gas = pd.read_csv(config_dir / "gas_subspace_summary.csv")
            gas.insert(0, "training_seed", seed)
            gas.insert(0, "dropout_rate", rate)
            gas_frames.append(gas)

            contrast = pd.read_csv(config_dir / "protocol_contrasts.csv")
            contrast.insert(0, "training_seed", seed)
            contrast.insert(0, "dropout_rate", rate)
            contrast_frames.append(contrast)

            gas_contrast = pd.read_csv(config_dir / "gas_subspace_contrasts.csv")
            gas_contrast.insert(0, "training_seed", seed)
            gas_contrast.insert(0, "dropout_rate", rate)
            gas_contrast_frames.append(gas_contrast)

            metrics = json.loads((config_dir / "model_metrics.json").read_text(encoding="utf-8"))
            model_rows.append(metrics)

    method_models = pd.concat(method_frames, ignore_index=True)
    gas_models = pd.concat(gas_frames, ignore_index=True)
    contrast_models = pd.concat(contrast_frames, ignore_index=True)
    gas_contrast_models = pd.concat(gas_contrast_frames, ignore_index=True)
    model_metrics = pd.DataFrame(model_rows)

    method_summary = aggregate_models(
        method_models,
        groups=["dropout_rate", "condition", "method"],
        metrics=[
            "kendall_tau_mean",
            "kendall_tau_relevant_mean",
            "l1_mean",
            "top5_recovery_mean",
        ],
    )
    gas_summary = aggregate_models(
        gas_models,
        groups=["dropout_rate", "condition"],
        metrics=[
            "retained_90_mean",
            "subspace_overlap_mean",
            "max_principal_angle_mean",
            "top_direction_cosine_mean",
        ],
    )
    model_fit_summary = aggregate_models(
        model_metrics,
        groups=["dropout_rate"],
        metrics=[
            "test_deterministic_r2",
            "test_mc_mean_r2",
            "mean_mc_predictive_sd",
            "prediction_sd",
        ],
    )
    contrast_summary = aggregate_models(
        contrast_models,
        groups=["dropout_rate", "method", "metric", "contrast"],
        metrics=["mean_difference"],
    )
    gas_contrast_summary = aggregate_models(
        gas_contrast_models,
        groups=["dropout_rate", "metric", "contrast"],
        metrics=["mean_difference"],
    )

    method_models.to_csv(out_dir / "method_model_level.csv", index=False)
    method_summary.to_csv(out_dir / "method_across_models.csv", index=False)
    gas_models.to_csv(out_dir / "gas_subspace_model_level.csv", index=False)
    gas_summary.to_csv(out_dir / "gas_subspace_across_models.csv", index=False)
    model_metrics.to_csv(out_dir / "model_fit_model_level.csv", index=False)
    model_fit_summary.to_csv(out_dir / "model_fit_across_models.csv", index=False)
    contrast_models.to_csv(out_dir / "protocol_contrasts_model_level.csv", index=False)
    contrast_summary.to_csv(out_dir / "protocol_contrasts_across_models.csv", index=False)
    gas_contrast_models.to_csv(out_dir / "gas_subspace_contrasts_model_level.csv", index=False)
    gas_contrast_summary.to_csv(out_dir / "gas_subspace_contrasts_across_models.csv", index=False)
    make_plot(method_summary, gas_summary, out_dir / "dropout_sweep_summary.png")

    manifest = {
        "training_seeds": seeds,
        "dropout_rates": rates,
        "sensitivity_repeats_per_model": repeats,
        "training_steps": steps,
        "number_of_trained_models": len(seeds) * len(rates),
        "statistical_unit": "independently trained neural network",
        "interval": "two-sided 95% Student-t interval across trained models",
    }
    (out_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nModel fit summary:")
    print(model_fit_summary.to_string(index=False))
    print("\nGAS method summary:")
    print(method_summary[method_summary.method == "GAS"].to_string(index=False))
    print("\nGAS subspace summary:")
    print(gas_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dropout-rates", type=float, nargs="+", default=DEFAULT_RATES)
    args = parser.parse_args()
    run_sweep(
        args.out,
        seeds=args.seeds,
        rates=args.dropout_rates,
        repeats=args.repeats,
        steps=args.steps,
    )
