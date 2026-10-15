"""End-to-end budget comparison for GAS-guided neural-network search.

The previous direction-intervention study held only the candidate budget fixed.
This study also charges the model calls used to estimate the CRN-GAS subspace.
Because a global subspace can be reused, the experiment reports several reuse
counts.  A reuse count of one is the conservative one-shot case; larger counts
amortize the single global GAS estimate across several optimization tasks.

For each matched budget, GAS receives a fixed paired candidate budget.  Random,
coordinate, and local-gradient search spend the budget they save by not fitting
the GAS subspace on additional paired candidates.  Candidate selection uses
paired-mask MC-dropout means and the chosen point is verified with the known
Friedman data-generating function.
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
from neural_network_gas_direction_intervention import (
    deterministic_gradient,
    directions_from_basis,
    mc_means_with_shared_masks,
    normalize_rows,
    paired_candidates,
)
from neural_network_gas_mc_dropout import gas_scores, train_dropout_mlp
from neural_network_gas_mc_dropout_sweep import aggregate_models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "end_to_end_budget"
DEFAULT_SEEDS = [2026101401, 2026101402, 2026101403, 2026101404, 2026101405]
METHODS = ["gas_crn", "random", "coordinate", "local_gradient"]
METHOD_LABELS = {
    "gas_crn": "CRN-GAS",
    "random": "Random directions",
    "coordinate": "Coordinate search",
    "local_gradient": "Local-gradient oracle",
}


def van_der_corput(index: int, base: int = 2) -> float:
    """Return one element of a base-b van der Corput sequence."""
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def nested_radius_fractions(n: int) -> np.ndarray:
    """A prefix-friendly radius schedule that covers the full trust interval."""
    if n <= 0:
        return np.empty(0)
    return np.asarray([1.0] + [van_der_corput(i) for i in range(1, n)], dtype=float)


def method_candidate_budget(
    *,
    method: str,
    gas_candidate_budget: int,
    gas_direction_evaluations: int,
    reuse_count: int,
    mc_passes: int,
    gradient_equivalent_evaluations: int,
) -> tuple[int, float, float, float]:
    """Return candidate budget and the matched per-task accounting quantities."""
    target = (
        gas_direction_evaluations / reuse_count
        + mc_passes * (1 + gas_candidate_budget)
    )
    if method == "gas_crn":
        return gas_candidate_budget, gas_direction_evaluations / reuse_count, target, 0.0

    direction_cost = (
        float(gradient_equivalent_evaluations) if method == "local_gradient" else 0.0
    )
    saved = max(0.0, gas_direction_evaluations / reuse_count - direction_cost)
    extra_pairs = int(math.floor(saved / (2 * mc_passes)))
    candidate_budget = gas_candidate_budget + 2 * extra_pairs
    used = direction_cost + mc_passes * (1 + candidate_budget)
    unused = target - used
    return candidate_budget, direction_cost, used, unused


def build_budget_table(
    *,
    gas_direction_evaluations: int,
    gas_candidate_budgets: list[int],
    reuse_counts: list[int],
    mc_passes: int,
    gradient_equivalent_evaluations: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for gas_budget in gas_candidate_budgets:
        for reuse_count in reuse_counts:
            target = (
                gas_direction_evaluations / reuse_count
                + mc_passes * (1 + gas_budget)
            )
            for method in METHODS:
                candidate_budget, direction_cost, used, unused = method_candidate_budget(
                    method=method,
                    gas_candidate_budget=gas_budget,
                    gas_direction_evaluations=gas_direction_evaluations,
                    reuse_count=reuse_count,
                    mc_passes=mc_passes,
                    gradient_equivalent_evaluations=gradient_equivalent_evaluations,
                )
                rows.append(
                    {
                        "method": method,
                        "reuse_count": reuse_count,
                        "gas_candidate_budget": gas_budget,
                        "method_candidate_budget": candidate_budget,
                        "direction_cost_per_task": direction_cost,
                        "target_total_evaluations_per_task": target,
                        "used_total_evaluations_per_task": used,
                        "unused_evaluations_per_task": unused,
                    }
                )
    return pd.DataFrame(rows)


def coordinate_directions(
    d: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    directions: list[np.ndarray] = []
    radii: list[float] = []
    cycles = int(math.ceil(n_pairs / d))
    cycle_radii = nested_radius_fractions(cycles)
    for cycle in range(cycles):
        for coordinate in rng.permutation(d):
            direction = np.zeros(d)
            direction[coordinate] = 1.0
            directions.append(direction)
            radii.append(float(cycle_radii[cycle]))
            if len(directions) == n_pairs:
                return np.asarray(directions), np.asarray(radii)
    raise RuntimeError("Failed to construct coordinate directions")


def build_candidate_pools(
    model,
    x: np.ndarray,
    *,
    gas_vectors: np.ndarray,
    gas_dimension: int,
    max_budget_by_method: dict[str, int],
    radius: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    d = len(x)
    pools: dict[str, np.ndarray] = {}

    gas_pairs = max_budget_by_method["gas_crn"] // 2
    gas_coefficients = rng.standard_normal((gas_pairs, gas_dimension))
    pools["gas_crn"] = paired_candidates(
        x,
        directions_from_basis(gas_vectors[:, :gas_dimension], gas_coefficients),
        np.full(gas_pairs, radius),
    )

    random_pairs = max_budget_by_method["random"] // 2
    pools["random"] = paired_candidates(
        x,
        normalize_rows(rng.standard_normal((random_pairs, d))),
        np.full(random_pairs, radius),
    )

    coordinate_pairs = max_budget_by_method["coordinate"] // 2
    coordinate_vectors, coordinate_radius_fractions = coordinate_directions(
        d, coordinate_pairs, rng
    )
    pools["coordinate"] = paired_candidates(
        x,
        coordinate_vectors,
        radius * coordinate_radius_fractions,
    )

    gradient_pairs = max_budget_by_method["local_gradient"] // 2
    gradient = deterministic_gradient(model, x)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm < 1e-12:
        gradient_direction = normalize_rows(rng.standard_normal((1, d)))[0]
    else:
        gradient_direction = gradient / gradient_norm
    gradient_vectors = np.repeat(gradient_direction[None, :], gradient_pairs, axis=0)
    pools["local_gradient"] = paired_candidates(
        x,
        gradient_vectors,
        radius * nested_radius_fractions(gradient_pairs),
    )
    return pools


def evaluate_budget_prefixes(
    model,
    x: np.ndarray,
    candidates: np.ndarray,
    *,
    budgets: list[int],
    mc_passes: int,
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    base_pred, candidate_pred = mc_means_with_shared_masks(
        model, x, candidates, mc_passes=mc_passes, rng=rng
    )
    base_true = float(friedman_function(x[None, :])[0])
    candidate_true = friedman_function(candidates)
    results: dict[int, dict[str, float]] = {}
    for budget in sorted(set(budgets)):
        predictions = candidate_pred[:budget]
        truths = candidate_true[:budget]
        chosen = int(np.argmin(predictions))
        predicted_improvement = max(0.0, base_pred - float(predictions[chosen]))
        selected_true_improvement = (
            base_true - float(truths[chosen]) if predicted_improvement > 0.0 else 0.0
        )
        results[budget] = {
            "predicted_improvement": predicted_improvement,
            "selected_true_improvement": selected_true_improvement,
            "oracle_true_improvement": max(0.0, base_true - float(np.min(truths))),
            "true_improvement_success": float(selected_true_improvement > 0.0),
        }
    return results


def paired_contrasts(model_level: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("gas_crn", "random", "gas_crn_minus_random"),
        ("gas_crn", "coordinate", "gas_crn_minus_coordinate"),
        ("local_gradient", "gas_crn", "gradient_minus_gas_crn"),
    ]
    metrics = [
        "mean_selected_true_improvement",
        "mean_oracle_true_improvement",
        "success_rate",
    ]
    rows: list[dict[str, object]] = []
    group_columns = ["reuse_count", "gas_candidate_budget"]
    for keys, subset in model_level.groupby(group_columns):
        reuse_count, gas_budget = keys
        for metric in metrics:
            pivot = subset.pivot(index="training_seed", columns="method", values=metric)
            for minuend, subtrahend, label in comparisons:
                differences = pivot[minuend] - pivot[subtrahend]
                for seed, difference in differences.items():
                    rows.append(
                        {
                            "training_seed": int(seed),
                            "reuse_count": int(reuse_count),
                            "gas_candidate_budget": int(gas_budget),
                            "metric": metric,
                            "contrast": label,
                            "difference": float(difference),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_break_even(contrast_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize when CRN-GAS first has a positive cross-model 95% interval."""
    primary = contrast_summary[
        (contrast_summary.metric == "mean_selected_true_improvement")
        & (contrast_summary.contrast == "gas_crn_minus_random")
    ].copy()
    rows: list[dict[str, float | int | None]] = []
    for gas_budget, subset in primary.groupby("gas_candidate_budget"):
        subset = subset.sort_values("reuse_count")
        conclusive = subset[subset.difference_t95_low > 0.0]
        one_shot = subset.iloc[0]
        rows.append(
            {
                "gas_candidate_budget": int(gas_budget),
                "one_shot_difference_mean": float(one_shot.difference_across_models_mean),
                "one_shot_t95_low": float(one_shot.difference_t95_low),
                "one_shot_t95_high": float(one_shot.difference_t95_high),
                "first_conclusive_reuse_count": (
                    int(conclusive.iloc[0].reuse_count) if len(conclusive) else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("gas_candidate_budget")


def make_plot(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    budget_table: pd.DataFrame,
    out: Path,
) -> None:
    colors = {
        "gas_crn": "#4F81BD",
        "random": "#8C8C8C",
        "coordinate": "#59A14F",
        "local_gradient": "#B07AA1",
    }
    max_gas_budget = int(summary.gas_candidate_budget.max())
    panel_summary = summary[summary.gas_candidate_budget == max_gas_budget]
    reuse_counts = sorted(panel_summary.reuse_count.unique().tolist())
    x_positions = [150 + i * 210 for i in range(len(reuse_counts))]

    image = Image.new("RGB", (1900, 1150), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 35),
        "End-to-end budget: when does a reusable GAS subspace pay off?",
        fill="#17365D",
        font=font(32, True),
    )
    draw.text(
        (72, 83),
        "All model calls are charged; selected points are verified with the true Friedman function.",
        fill="#666666",
        font=font(17),
    )
    draw.text(
        (72, 113),
        "The local-gradient oracle is charged as two forward-equivalent evaluations per starting point.",
        fill="#777777",
        font=font(15),
    )

    # Panel 1: absolute achieved improvement.
    x0, y0, width, height = 110, 220, 950, 610
    draw.text((x0, 165), f"Matched-budget true improvement (GAS candidates = {max_gas_budget})", fill="#17365D", font=font(21, True))
    draw.line((x0, y0, x0, y0 + height), fill="#333333", width=2)
    draw.line((x0, y0 + height, x0 + width, y0 + height), fill="#333333", width=2)
    y_max = float(panel_summary.mean_selected_true_improvement_across_models_mean.max()) * 1.15
    for tick in range(6):
        value = y_max * tick / 5
        yy = y0 + height - int(height * tick / 5)
        draw.line((x0 - 6, yy, x0, yy), fill="#333333", width=2)
        draw.text((x0 - 78, yy - 10), f"{value:.2f}", fill="#666666", font=font(14))
    for xx, reuse in zip(x_positions, reuse_counts):
        draw.text((xx - 15, y0 + height + 14), str(reuse), fill="#222222", font=font(15))
    draw.text((430, y0 + height + 55), "number of tasks reusing one GAS estimate", fill="#666666", font=font(15))
    for method in METHODS:
        method_rows = panel_summary[panel_summary.method == method].set_index("reuse_count")
        points: list[tuple[int, int]] = []
        for xx, reuse in zip(x_positions, reuse_counts):
            value = float(method_rows.loc[reuse, "mean_selected_true_improvement_across_models_mean"])
            yy = y0 + height - int(height * value / y_max)
            low = float(method_rows.loc[reuse, "mean_selected_true_improvement_t95_low"])
            high = float(method_rows.loc[reuse, "mean_selected_true_improvement_t95_high"])
            low_y = y0 + height - int(height * low / y_max)
            high_y = y0 + height - int(height * high / y_max)
            draw.line((xx, high_y, xx, low_y), fill=colors[method], width=2)
            draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=colors[method])
            points.append((xx, yy))
        draw.line(points, fill=colors[method], width=4)

    # Panel 2: paired GAS-minus-random contrast.
    x1, y1, w1, h1 = 1190, 220, 590, 360
    draw.text((x1, 165), "CRN-GAS minus random", fill="#17365D", font=font(21, True))
    draw.line((x1, y1, x1, y1 + h1), fill="#333333", width=2)
    draw.line((x1, y1 + h1, x1 + w1, y1 + h1), fill="#333333", width=2)
    contrast_rows = contrasts[
        (contrasts.gas_candidate_budget == max_gas_budget)
        & (contrasts.metric == "mean_selected_true_improvement")
        & (contrasts.contrast == "gas_crn_minus_random")
    ].set_index("reuse_count")
    lower = min(-0.5, float(contrast_rows.difference_t95_low.min()) * 1.15)
    upper = max(0.5, float(contrast_rows.difference_t95_high.max()) * 1.15)
    zero_y = y1 + int(h1 * upper / (upper - lower))
    draw.line((x1, zero_y, x1 + w1, zero_y), fill="#999999", width=2)
    for tick in range(5):
        value = lower + (upper - lower) * tick / 4
        yy = y1 + h1 - int(h1 * tick / 4)
        draw.line((x1 - 6, yy, x1, yy), fill="#333333", width=2)
        draw.text((x1 - 72, yy - 10), f"{value:.2f}", fill="#666666", font=font(13))
    panel2_x = [x1 + 55 + i * 120 for i in range(len(reuse_counts))]
    points = []
    for xx, reuse in zip(panel2_x, reuse_counts):
        row = contrast_rows.loc[reuse]
        mean = float(row.difference_across_models_mean)
        low = float(row.difference_t95_low)
        high = float(row.difference_t95_high)
        transform = lambda value: y1 + int(h1 * (upper - value) / (upper - lower))
        yy, low_y, high_y = transform(mean), transform(low), transform(high)
        draw.line((xx, high_y, xx, low_y), fill="#4F81BD", width=3)
        draw.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill="#4F81BD")
        points.append((xx, yy))
        draw.text((xx - 15, y1 + h1 + 12), str(reuse), fill="#222222", font=font(14))
    draw.line(points, fill="#4F81BD", width=4)
    draw.text((x1 + 225, y1 + h1 + 52), "reuse count", fill="#666666", font=font(14))

    # Panel 3: candidate allocation explains the comparison.
    x2, y2 = 1190, 685
    draw.text((x2, y2), "Candidate allocation at the largest GAS budget", fill="#17365D", font=font(19, True))
    allocation = budget_table[
        (budget_table.gas_candidate_budget == max_gas_budget)
        & (budget_table.method.isin(["gas_crn", "random"]))
    ]
    gas_alloc = allocation[allocation.method == "gas_crn"].set_index("reuse_count")
    random_alloc = allocation[allocation.method == "random"].set_index("reuse_count")
    for index, reuse in enumerate(reuse_counts):
        yy = y2 + 55 + index * 47
        gas_n = int(gas_alloc.loc[reuse, "method_candidate_budget"])
        random_n = int(random_alloc.loc[reuse, "method_candidate_budget"])
        draw.text((x2, yy), f"reuse {reuse:>3}: GAS {gas_n:>3} candidates, random {random_n:>3}", fill="#333333", font=font(16))

    legend_y = 1080
    for index, method in enumerate(METHODS):
        xx = 180 + index * 420
        draw.line((xx, legend_y, xx + 45, legend_y), fill=colors[method], width=5)
        draw.text((xx + 58, legend_y - 12), METHOD_LABELS[method], fill="#222222", font=font(15))
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
    gas_m: int = 64,
    gas_q: int = 4,
    gas_candidate_budgets: list[int] | None = None,
    reuse_counts: list[int] | None = None,
    gradient_equivalent_evaluations: int = 2,
) -> None:
    if gas_candidate_budgets is None:
        gas_candidate_budgets = [4, 8, 12, 20]
    if reuse_counts is None:
        reuse_counts = [1, 4, 16, 64, 256]
    if any(value <= 0 or value % 2 for value in gas_candidate_budgets):
        raise ValueError("GAS candidate budgets must be positive even integers")
    if any(value <= 0 for value in reuse_counts):
        raise ValueError("Reuse counts must be positive")
    if mc_passes <= 0 or gas_m <= 0 or gas_q <= 0:
        raise ValueError("MC passes, GAS m, and GAS q must be positive")

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    gas_direction_evaluations = gas_m * (1 + 2 * gas_q * 10)
    budget_table = build_budget_table(
        gas_direction_evaluations=gas_direction_evaluations,
        gas_candidate_budgets=gas_candidate_budgets,
        reuse_counts=reuse_counts,
        mc_passes=mc_passes,
        gradient_equivalent_evaluations=gradient_equivalent_evaluations,
    )
    max_budget_by_method = {
        method: int(budget_table[budget_table.method == method].method_candidate_budget.max())
        for method in METHODS
    }
    if max(max_budget_by_method.values()) > 1000:
        raise ValueError("Configuration generates more than 1000 candidates per method")

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for model_index, training_seed in enumerate(training_seeds, start=1):
        print(f"[{model_index}/{len(training_seeds)}] training_seed={training_seed}", flush=True)
        model, model_metrics = train_dropout_mlp(
            seed=training_seed,
            steps=training_steps,
            dropout_rate=dropout_rate,
        )
        _, reported_evaluations, retained, eigenvalues, vectors = gas_scores(
            model,
            m=gas_m,
            q=gas_q,
            seed=training_seed + 200,
            stochastic=True,
            crn=True,
        )
        if reported_evaluations != gas_direction_evaluations:
            raise RuntimeError("GAS evaluation accounting mismatch")
        diagnostics.append(
            {
                "training_seed": training_seed,
                "gas_retained_dimension": retained,
                "gas_direction_evaluations": reported_evaluations,
                "leading_eigenvalue_fraction": float(eigenvalues[0] / eigenvalues.sum()),
                "test_mc_mean_r2": model_metrics["test_mc_mean_r2"],
            }
        )

        start_rng = np.random.default_rng(training_seed + 300)
        starting_points = start_rng.uniform(0.2, 0.8, size=(n_starts, 10))
        for start_index, x in enumerate(starting_points):
            direction_rng = np.random.default_rng(training_seed + 100000 + start_index)
            pools = build_candidate_pools(
                model,
                x,
                gas_vectors=vectors,
                gas_dimension=retained,
                max_budget_by_method=max_budget_by_method,
                radius=radius,
                rng=direction_rng,
            )
            method_results: dict[str, dict[int, dict[str, float]]] = {}
            for method in METHODS:
                method_budgets = sorted(
                    budget_table[budget_table.method == method]
                    .method_candidate_budget.astype(int)
                    .unique()
                    .tolist()
                )
                evaluation_rng = np.random.default_rng(
                    training_seed + 500000 + 1000 * start_index
                )
                method_results[method] = evaluate_budget_prefixes(
                    model,
                    x,
                    pools[method],
                    budgets=method_budgets,
                    mc_passes=mc_passes,
                    rng=evaluation_rng,
                )

            for budget_row in budget_table.itertuples(index=False):
                result = method_results[budget_row.method][budget_row.method_candidate_budget]
                rows.append(
                    {
                        "training_seed": training_seed,
                        "start_index": start_index,
                        "method": budget_row.method,
                        "reuse_count": int(budget_row.reuse_count),
                        "gas_candidate_budget": int(budget_row.gas_candidate_budget),
                        "method_candidate_budget": int(budget_row.method_candidate_budget),
                        "direction_cost_per_task": float(budget_row.direction_cost_per_task),
                        "target_total_evaluations_per_task": float(budget_row.target_total_evaluations_per_task),
                        "used_total_evaluations_per_task": float(budget_row.used_total_evaluations_per_task),
                        "unused_evaluations_per_task": float(budget_row.unused_evaluations_per_task),
                        **result,
                    }
                )

    by_start = pd.DataFrame(rows)
    diagnostic_df = pd.DataFrame(diagnostics)
    group_columns = [
        "training_seed",
        "method",
        "reuse_count",
        "gas_candidate_budget",
        "method_candidate_budget",
        "target_total_evaluations_per_task",
    ]
    model_level = (
        by_start.groupby(group_columns, as_index=False)
        .agg(
            mean_selected_true_improvement=("selected_true_improvement", "mean"),
            mean_predicted_improvement=("predicted_improvement", "mean"),
            mean_oracle_true_improvement=("oracle_true_improvement", "mean"),
            success_rate=("true_improvement_success", "mean"),
        )
    )
    across_models = aggregate_models(
        model_level,
        groups=[
            "method",
            "reuse_count",
            "gas_candidate_budget",
            "method_candidate_budget",
            "target_total_evaluations_per_task",
        ],
        metrics=[
            "mean_selected_true_improvement",
            "mean_predicted_improvement",
            "mean_oracle_true_improvement",
            "success_rate",
        ],
    )
    contrast_models = paired_contrasts(model_level)
    contrast_summary = aggregate_models(
        contrast_models,
        groups=["reuse_count", "gas_candidate_budget", "metric", "contrast"],
        metrics=["difference"],
    )
    break_even = summarize_break_even(contrast_summary)

    by_start.to_csv(out_dir / "end_to_end_by_start.csv", index=False)
    budget_table.to_csv(out_dir / "budget_accounting.csv", index=False)
    diagnostic_df.to_csv(out_dir / "gas_diagnostics.csv", index=False)
    model_level.to_csv(out_dir / "end_to_end_model_level.csv", index=False)
    across_models.to_csv(out_dir / "end_to_end_across_models.csv", index=False)
    contrast_models.to_csv(out_dir / "end_to_end_contrasts_model_level.csv", index=False)
    contrast_summary.to_csv(out_dir / "end_to_end_contrasts_across_models.csv", index=False)
    break_even.to_csv(out_dir / "break_even_summary.csv", index=False)
    make_plot(
        across_models,
        contrast_summary,
        budget_table,
        out_dir / "end_to_end_budget_summary.png",
    )

    metadata = {
        "training_seeds": training_seeds,
        "dropout_rate": dropout_rate,
        "training_steps": training_steps,
        "starting_points_per_model": n_starts,
        "mc_passes_per_candidate": mc_passes,
        "trust_region_radius": radius,
        "gas_m": gas_m,
        "gas_q": gas_q,
        "gas_direction_evaluations": gas_direction_evaluations,
        "gas_candidate_budgets": gas_candidate_budgets,
        "reuse_counts": reuse_counts,
        "gradient_equivalent_evaluations_per_start": gradient_equivalent_evaluations,
        "input_starting_region": "independent Uniform(0.2, 0.8)",
        "selection_objective": "paired-mask Monte Carlo mean prediction",
        "verification_objective": "known Friedman data-generating function",
        "statistical_unit": "independently trained network",
        "budget_rule": "GAS direction cost is divided by reuse count; saved calls fund additional paired candidates for comparators",
        "gas_dimension_rule": "90% activity-eigenvalue energy from the charged low-budget CRN-GAS estimate",
        "runtime_seconds": time.perf_counter() - started,
    }
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    largest_budget = max(gas_candidate_budgets)
    selected = contrast_summary[
        (contrast_summary.gas_candidate_budget == largest_budget)
        & (contrast_summary.metric == "mean_selected_true_improvement")
        & (contrast_summary.contrast == "gas_crn_minus_random")
    ][
        [
            "reuse_count",
            "difference_across_models_mean",
            "difference_t95_low",
            "difference_t95_high",
        ]
    ]
    print("\nCRN-GAS minus random at the largest GAS candidate budget:")
    print(selected.sort_values("reuse_count").to_string(index=False))
    print("\nMaximum candidate budgets:")
    print(json.dumps(max_budget_by_method, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dropout-rate", type=float, default=0.10)
    parser.add_argument("--training-steps", type=int, default=3500)
    parser.add_argument("--starts", type=int, default=256)
    parser.add_argument("--mc-passes", type=int, default=32)
    parser.add_argument("--radius", type=float, default=0.15)
    parser.add_argument("--gas-m", type=int, default=64)
    parser.add_argument("--gas-q", type=int, default=4)
    parser.add_argument("--gas-candidate-budgets", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--reuse-counts", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--gradient-equivalent-evaluations", type=int, default=2)
    args = parser.parse_args()
    run(
        args.out,
        training_seeds=args.training_seeds,
        dropout_rate=args.dropout_rate,
        training_steps=args.training_steps,
        n_starts=args.starts,
        mc_passes=args.mc_passes,
        radius=args.radius,
        gas_m=args.gas_m,
        gas_q=args.gas_q,
        gas_candidate_budgets=args.gas_candidate_budgets,
        reuse_counts=args.reuse_counts,
        gradient_equivalent_evaluations=args.gradient_equivalent_evaluations,
    )
