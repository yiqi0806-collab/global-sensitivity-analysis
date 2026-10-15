"""Test whether GAS activity directions are useful for black-box intervention.

For five independently trained 10%-dropout ReLU networks, this experiment:

1. estimates a high-budget deterministic GAS reference subspace;
2. estimates low-budget GAS subspaces with paired and independent dropout masks;
3. searches around held-out starting points at matched candidate-evaluation
   budgets using GAS, random, coordinate, inactive, and local-gradient directions;
4. selects candidates using a paired-mask Monte Carlo objective and verifies the
   selected improvement against the known Friedman data-generating function.

The direction-estimation cost is reported separately.  The intervention budget
counts candidate evaluations only, so this study measures conditional direction
quality rather than end-to-end cost efficiency.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from neural_network_gas_demo import font, friedman_function
from neural_network_gas_mc_dropout import (
    DropoutMLP,
    gas_scores,
    subspace_metrics,
    train_dropout_mlp,
)
from neural_network_gas_mc_dropout_sweep import aggregate_models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "direction_intervention"
DEFAULT_SEEDS = [2026101401, 2026101402, 2026101403, 2026101404, 2026101405]
METHODS = [
    "gas_reference",
    "gas_crn",
    "gas_independent",
    "random",
    "coordinate",
    "local_gradient",
    "inactive",
]
METHOD_LABELS = {
    "gas_reference": "GAS reference",
    "gas_crn": "GAS paired masks",
    "gas_independent": "GAS independent masks",
    "random": "Random directions",
    "coordinate": "Coordinate directions",
    "local_gradient": "Local gradient oracle",
    "inactive": "Inactive subspace",
}


def deterministic_gradient(model: DropoutMLP, x: np.ndarray) -> np.ndarray:
    """Gradient of the dropout-disabled network output with respect to x."""
    xs = (np.asarray(x, dtype=float) - 0.5) / math.sqrt(1.0 / 12.0)
    z1 = xs @ model.w1 + model.b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ model.w2 + model.b2
    grad_a2 = model.w3[:, 0] * model.y_std
    grad_z2 = grad_a2 * (z2 > 0.0)
    grad_a1 = grad_z2 @ model.w2.T
    grad_z1 = grad_a1 * (z1 > 0.0)
    return (grad_z1 @ model.w1.T) / math.sqrt(1.0 / 12.0)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def directions_from_basis(
    basis: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    return normalize_rows(coefficients @ basis.T)


def paired_candidates(
    x: np.ndarray,
    directions: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    candidates = np.empty((2 * len(directions), len(x)))
    candidates[0::2] = x[None, :] + radii[:, None] * directions
    candidates[1::2] = x[None, :] - radii[:, None] * directions
    return np.clip(candidates, 0.0, 1.0)


def mc_means_with_shared_masks(
    model: DropoutMLP,
    x: np.ndarray,
    candidates: np.ndarray,
    *,
    mc_passes: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    masks = model.sample_masks(mc_passes, rng)
    base_inputs = np.repeat(x[None, :], mc_passes, axis=0)
    base_value = float(model.predict(base_inputs, masks).mean())

    repeated_candidates = np.repeat(candidates, mc_passes, axis=0)
    tiled_masks = (
        np.tile(masks[0], (len(candidates), 1)),
        np.tile(masks[1], (len(candidates), 1)),
    )
    values = model.predict(repeated_candidates, tiled_masks).reshape(
        len(candidates), mc_passes
    )
    return base_value, values.mean(axis=1)


def build_direction_sets(
    model: DropoutMLP,
    x: np.ndarray,
    *,
    reference_vectors: np.ndarray,
    crn_vectors: np.ndarray,
    independent_vectors: np.ndarray,
    dimension: int,
    max_pairs: int,
    radius: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    d = len(x)
    coefficients = rng.standard_normal((max_pairs, dimension))
    full_coefficients = rng.standard_normal((max_pairs, d))
    inactive_dimension = d - dimension
    inactive_coefficients = rng.standard_normal((max_pairs, inactive_dimension))

    coordinate_order = rng.permutation(d)[:max_pairs]
    coordinate_directions = np.eye(d)[coordinate_order]
    gradient = deterministic_gradient(model, x)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm < 1e-12:
        gradient_direction = normalize_rows(rng.standard_normal((1, d)))[0]
    else:
        gradient_direction = gradient / gradient_norm
    gradient_directions = np.repeat(gradient_direction[None, :], max_pairs, axis=0)
    gradient_radii = np.linspace(radius, radius / max_pairs, max_pairs)
    fixed_radii = np.full(max_pairs, radius)

    return {
        "gas_reference": paired_candidates(
            x,
            directions_from_basis(reference_vectors[:, :dimension], coefficients),
            fixed_radii,
        ),
        "gas_crn": paired_candidates(
            x,
            directions_from_basis(crn_vectors[:, :dimension], coefficients),
            fixed_radii,
        ),
        "gas_independent": paired_candidates(
            x,
            directions_from_basis(independent_vectors[:, :dimension], coefficients),
            fixed_radii,
        ),
        "random": paired_candidates(
            x, normalize_rows(full_coefficients), fixed_radii
        ),
        "coordinate": paired_candidates(x, coordinate_directions, fixed_radii),
        "local_gradient": paired_candidates(
            x, gradient_directions, gradient_radii
        ),
        "inactive": paired_candidates(
            x,
            directions_from_basis(
                reference_vectors[:, dimension:], inactive_coefficients
            ),
            fixed_radii,
        ),
    }


def evaluate_candidates(
    model: DropoutMLP,
    x: np.ndarray,
    candidates: np.ndarray,
    *,
    budgets: list[int],
    mc_passes: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    base_pred, candidate_pred = mc_means_with_shared_masks(
        model, x, candidates, mc_passes=mc_passes, rng=rng
    )
    base_true = float(friedman_function(x[None, :])[0])
    candidate_true = friedman_function(candidates)
    rows: list[dict[str, float]] = []
    for budget in budgets:
        pred = candidate_pred[:budget]
        truth = candidate_true[:budget]
        chosen = int(np.argmin(pred))
        predicted_improvement = max(0.0, base_pred - float(pred[chosen]))
        if predicted_improvement > 0.0:
            selected_true_improvement = base_true - float(truth[chosen])
        else:
            selected_true_improvement = 0.0
        oracle_true_improvement = max(0.0, base_true - float(np.min(truth)))
        max_abs_true_change = float(np.max(np.abs(truth - base_true)))
        rows.append(
            {
                "budget": budget,
                "baseline_predicted_value": base_pred,
                "baseline_true_value": base_true,
                "predicted_improvement": predicted_improvement,
                "selected_true_improvement": selected_true_improvement,
                "oracle_true_improvement": oracle_true_improvement,
                "max_abs_true_change": max_abs_true_change,
                "true_improvement_success": float(selected_true_improvement > 0.0),
            }
        )
    return rows


def paired_method_contrasts(model_level: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("gas_crn", "random", "gas_crn_minus_random"),
        ("gas_reference", "random", "gas_reference_minus_random"),
        ("gas_independent", "random", "gas_independent_minus_random"),
        ("gas_crn", "inactive", "gas_crn_minus_inactive"),
        ("local_gradient", "gas_crn", "gradient_minus_gas_crn"),
    ]
    metrics = [
        "mean_selected_true_improvement",
        "mean_oracle_true_improvement",
        "mean_max_abs_true_change",
        "success_rate",
    ]
    rows: list[dict[str, object]] = []
    for budget, subset in model_level.groupby("budget"):
        for metric in metrics:
            pivot = subset.pivot(index="training_seed", columns="method", values=metric)
            for minuend, subtrahend, label in comparisons:
                differences = pivot[minuend] - pivot[subtrahend]
                for seed, difference in differences.items():
                    rows.append(
                        {
                            "training_seed": int(seed),
                            "budget": int(budget),
                            "metric": metric,
                            "contrast": label,
                            "difference": float(difference),
                        }
                    )
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, out: Path) -> None:
    colors = {
        "gas_reference": "#17365D",
        "gas_crn": "#4F81BD",
        "gas_independent": "#F28E2B",
        "random": "#8C8C8C",
        "coordinate": "#59A14F",
        "local_gradient": "#B07AA1",
        "inactive": "#D62728",
    }
    panels = [
        ("mean_selected_true_improvement", "Selected true improvement"),
        ("mean_oracle_true_improvement", "Best available true improvement"),
        ("mean_max_abs_true_change", "Maximum absolute true change"),
    ]
    budgets = sorted(summary.budget.unique().tolist())
    image = Image.new("RGB", (2000, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 35),
        "Do GAS activity directions improve constrained search?",
        fill="#17365D",
        font=font(34, True),
    )
    draw.text(
        (72, 85),
        "Five trained 10%-dropout networks; objective selection uses paired-mask MC means.",
        fill="#666666",
        font=font(18),
    )

    for panel_index, (metric, title) in enumerate(panels):
        x0 = 100 + panel_index * 640
        y0, width, height = 220, 510, 580
        draw.text((x0, 160), title, fill="#17365D", font=font(20, True))
        draw.line((x0, y0, x0, y0 + height), fill="#333333", width=2)
        draw.line((x0, y0 + height, x0 + width, y0 + height), fill="#333333", width=2)
        metric_column = f"{metric}_across_models_mean"
        y_max = float(summary[metric_column].max()) * 1.12
        y_max = max(y_max, 0.1)
        for tick in range(6):
            value = y_max * tick / 5
            yy = y0 + height - int(height * tick / 5)
            draw.line((x0 - 6, yy, x0, yy), fill="#333333", width=2)
            draw.text((x0 - 78, yy - 10), f"{value:.2f}", fill="#666666", font=font(14))
        x_positions = [x0 + int(width * (i + 0.5) / len(budgets)) for i in range(len(budgets))]
        for xx, budget in zip(x_positions, budgets):
            draw.text((xx - 10, y0 + height + 14), str(budget), fill="#222222", font=font(16))
        draw.text((x0 + width // 2 - 55, y0 + height + 50), "candidate budget", fill="#666666", font=font(15))
        for method in METHODS:
            subset = summary[summary.method == method].set_index("budget")
            points: list[tuple[int, int]] = []
            for xx, budget in zip(x_positions, budgets):
                value = float(subset.loc[budget, metric_column])
                yy = y0 + height - int(height * value / y_max)
                points.append((xx, yy))
                draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=colors[method])
            draw.line(points, fill=colors[method], width=4)

    legend_y = 930
    for index, method in enumerate(METHODS):
        row = index // 4
        col = index % 4
        xx = 250 + col * 430
        yy = legend_y + row * 42
        draw.line((xx, yy, xx + 45, yy), fill=colors[method], width=5)
        draw.text((xx + 58, yy - 12), METHOD_LABELS[method], fill="#222222", font=font(16))
    image.save(out)


def run(
    out_dir: Path,
    *,
    training_seeds: list[int],
    dropout_rate: float = 0.10,
    training_steps: int = 3500,
    n_starts: int = 256,
    mc_passes: int = 32,
    radius: float = 0.15,
    budgets: list[int] | None = None,
) -> None:
    if budgets is None:
        budgets = [2, 4, 8, 12, 20]
    if any(budget <= 0 or budget % 2 for budget in budgets):
        raise ValueError("All candidate budgets must be positive even integers")
    max_budget = max(budgets)
    max_pairs = max_budget // 2
    if max_pairs > 10:
        raise ValueError("This 10-dimensional experiment supports at most 20 candidates")

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    start_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for model_index, training_seed in enumerate(training_seeds, start=1):
        print(
            f"[{model_index}/{len(training_seeds)}] training_seed={training_seed}",
            flush=True,
        )
        model, model_metrics = train_dropout_mlp(
            seed=training_seed,
            steps=training_steps,
            dropout_rate=dropout_rate,
        )
        (
            _,
            reference_evaluations,
            reference_dimension,
            reference_eigenvalues,
            reference_vectors,
        ) = gas_scores(
            model,
            m=4096,
            q=8,
            seed=training_seed + 100,
            stochastic=False,
            crn=True,
        )
        _, crn_evaluations, crn_retained, _, crn_vectors = gas_scores(
            model,
            m=64,
            q=4,
            seed=training_seed + 200,
            stochastic=True,
            crn=True,
        )
        _, independent_evaluations, independent_retained, _, independent_vectors = gas_scores(
            model,
            m=64,
            q=4,
            seed=training_seed + 200,
            stochastic=True,
            crn=False,
        )
        crn_overlap, crn_angle, crn_cosine = subspace_metrics(
            reference_vectors, crn_vectors, reference_dimension
        )
        ind_overlap, ind_angle, ind_cosine = subspace_metrics(
            reference_vectors, independent_vectors, reference_dimension
        )
        diagnostics.extend(
            [
                {
                    "training_seed": training_seed,
                    "method": "gas_crn",
                    "reference_dimension": reference_dimension,
                    "estimated_retained_90": crn_retained,
                    "subspace_overlap": crn_overlap,
                    "max_principal_angle": crn_angle,
                    "top_direction_cosine": crn_cosine,
                    "direction_estimation_evaluations": crn_evaluations,
                    "test_mc_mean_r2": model_metrics["test_mc_mean_r2"],
                },
                {
                    "training_seed": training_seed,
                    "method": "gas_independent",
                    "reference_dimension": reference_dimension,
                    "estimated_retained_90": independent_retained,
                    "subspace_overlap": ind_overlap,
                    "max_principal_angle": ind_angle,
                    "top_direction_cosine": ind_cosine,
                    "direction_estimation_evaluations": independent_evaluations,
                    "test_mc_mean_r2": model_metrics["test_mc_mean_r2"],
                },
                {
                    "training_seed": training_seed,
                    "method": "gas_reference",
                    "reference_dimension": reference_dimension,
                    "estimated_retained_90": reference_dimension,
                    "subspace_overlap": 1.0,
                    "max_principal_angle": 0.0,
                    "top_direction_cosine": 1.0,
                    "direction_estimation_evaluations": reference_evaluations,
                    "test_mc_mean_r2": model_metrics["test_mc_mean_r2"],
                },
            ]
        )

        rng = np.random.default_rng(training_seed + 300)
        starting_points = rng.uniform(0.2, 0.8, size=(n_starts, 10))
        for start_index, x in enumerate(starting_points):
            direction_rng = np.random.default_rng(training_seed + 100000 + start_index)
            candidate_sets = build_direction_sets(
                model,
                x,
                reference_vectors=reference_vectors,
                crn_vectors=crn_vectors,
                independent_vectors=independent_vectors,
                dimension=reference_dimension,
                max_pairs=max_pairs,
                radius=radius,
                rng=direction_rng,
            )
            for method in METHODS:
                evaluation_rng = np.random.default_rng(
                    training_seed + 500000 + 1000 * start_index
                )
                result_rows = evaluate_candidates(
                    model,
                    x,
                    candidate_sets[method],
                    budgets=budgets,
                    mc_passes=mc_passes,
                    rng=evaluation_rng,
                )
                for result in result_rows:
                    start_rows.append(
                        {
                            "training_seed": training_seed,
                            "start_index": start_index,
                            "method": method,
                            **result,
                        }
                    )

    starts = pd.DataFrame(start_rows)
    diagnostic_df = pd.DataFrame(diagnostics)
    model_level = (
        starts.groupby(["training_seed", "method", "budget"], as_index=False)
        .agg(
            mean_selected_true_improvement=("selected_true_improvement", "mean"),
            mean_predicted_improvement=("predicted_improvement", "mean"),
            mean_oracle_true_improvement=("oracle_true_improvement", "mean"),
            mean_max_abs_true_change=("max_abs_true_change", "mean"),
            success_rate=("true_improvement_success", "mean"),
        )
    )
    across_models = aggregate_models(
        model_level,
        groups=["method", "budget"],
        metrics=[
            "mean_selected_true_improvement",
            "mean_predicted_improvement",
            "mean_oracle_true_improvement",
            "mean_max_abs_true_change",
            "success_rate",
        ],
    )
    contrast_models = paired_method_contrasts(model_level)
    contrast_summary = aggregate_models(
        contrast_models,
        groups=["budget", "metric", "contrast"],
        metrics=["difference"],
    )

    starts.to_csv(out_dir / "intervention_by_start.csv", index=False)
    model_level.to_csv(out_dir / "intervention_model_level.csv", index=False)
    across_models.to_csv(out_dir / "intervention_across_models.csv", index=False)
    diagnostic_df.to_csv(out_dir / "direction_diagnostics.csv", index=False)
    contrast_models.to_csv(out_dir / "method_contrasts_model_level.csv", index=False)
    contrast_summary.to_csv(out_dir / "method_contrasts_across_models.csv", index=False)
    make_plot(across_models, out_dir / "direction_intervention_summary.png")

    metadata = {
        "training_seeds": training_seeds,
        "dropout_rate": dropout_rate,
        "training_steps": training_steps,
        "starting_points_per_model": n_starts,
        "mc_passes_per_candidate": mc_passes,
        "trust_region_radius": radius,
        "candidate_budgets": budgets,
        "input_starting_region": "independent Uniform(0.2, 0.8)",
        "selection_objective": "paired-mask Monte Carlo mean prediction",
        "verification_objective": "known Friedman data-generating function",
        "statistical_unit": "independently trained network",
        "direction_estimation_cost_included_in_candidate_budget": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    selected = across_models[
        across_models.budget == max_budget
    ][
        [
            "method",
            "mean_selected_true_improvement_across_models_mean",
            "mean_oracle_true_improvement_across_models_mean",
            "mean_max_abs_true_change_across_models_mean",
            "success_rate_across_models_mean",
        ]
    ]
    print("\nBudget-20 result:")
    print(selected.sort_values("mean_selected_true_improvement_across_models_mean", ascending=False).to_string(index=False))
    print("\nDirection diagnostics:")
    print(diagnostic_df.groupby("method").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dropout-rate", type=float, default=0.10)
    parser.add_argument("--training-steps", type=int, default=3500)
    parser.add_argument("--starts", type=int, default=256)
    parser.add_argument("--mc-passes", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 12, 20])
    args = parser.parse_args()
    run(
        args.out,
        training_seeds=args.training_seeds,
        dropout_rate=args.dropout_rate,
        training_steps=args.training_steps,
        n_starts=args.starts,
        mc_passes=args.mc_passes,
        radius=args.radius,
        budgets=args.budgets,
    )
