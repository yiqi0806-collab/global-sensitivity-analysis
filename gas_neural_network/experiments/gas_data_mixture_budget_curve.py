"""Budget curve for GAS direction estimation on a data-mixture objective.

This follow-up keeps the six-source data, MLP, simplex parameterization, and
validation protocol from ``gas_data_mixture_pilot.py`` fixed.  Only the number
of outer GAS design points changes.  Nested designs make the budget comparison
efficient and ensure that a larger budget contains every smaller-budget point.

The experiment separates two questions:

1. Does additional direction-estimation budget improve recovery of the
   high-budget leading activity direction under CRN and independent training
   randomness?
2. Does that improved estimate translate into better verified mixture search?
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

from gas_data_mixture_pilot import (
    MixtureObjective,
    SOURCE_NAMES,
    activity_from_objective,
    eigenvectors_from_activity,
    font,
    generate_data,
    mean_objective,
    mixture_from_u,
    t_interval,
)
from neural_network_gas_direction_intervention import normalize_rows
from neural_network_gas_mc_dropout import subspace_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "data_mixture_budget_curve"


def nested_activity_estimates(
    objective: MixtureObjective,
    *,
    m_values: list[int],
    q: int,
    seed: int,
    crn: bool,
) -> dict[int, dict[str, object]]:
    """Estimate activity matrices from nested prefixes of one common design."""
    m_values = sorted(set(m_values))
    max_m = max(m_values)
    d = len(SOURCE_NAMES) - 1
    rng = np.random.default_rng(seed)
    points = rng.random((max_m, d))
    base_seeds = rng.integers(1, 2**31 - 1, size=max_m)
    base_values = np.asarray(
        [objective(point, int(run_seed)) for point, run_seed in zip(points, base_seeds)]
    )
    nodes_raw, weights_raw = np.polynomial.legendre.leggauss(q)
    nodes = 0.5 * (nodes_raw + 1.0)
    weights = 0.5 * weights_raw
    row_contributions = np.zeros((max_m, d, d))

    for row in range(max_m):
        x = points[row]
        averaged_slopes = np.zeros(d)
        squared_slopes = np.zeros(d)
        for coordinate in range(d):
            left_points = np.repeat(x[None, :], q, axis=0)
            right_points = np.repeat(x[None, :], q, axis=0)
            left_points[:, coordinate] = x[coordinate] * nodes
            right_points[:, coordinate] = x[coordinate] + (1.0 - x[coordinate]) * nodes
            if crn:
                left_seeds = np.repeat(base_seeds[row], q)
                right_seeds = np.repeat(base_seeds[row], q)
            else:
                left_seeds = rng.integers(1, 2**31 - 1, size=q)
                right_seeds = rng.integers(1, 2**31 - 1, size=q)
            left_values = np.asarray(
                [objective(point, int(run_seed)) for point, run_seed in zip(left_points, left_seeds)]
            )
            right_values = np.asarray(
                [objective(point, int(run_seed)) for point, run_seed in zip(right_points, right_seeds)]
            )
            left_slopes = (left_values - base_values[row]) / (
                left_points[:, coordinate] - x[coordinate]
            )
            right_slopes = (right_values - base_values[row]) / (
                right_points[:, coordinate] - x[coordinate]
            )
            averaged_slopes[coordinate] = (
                x[coordinate] * np.sum(weights * left_slopes)
                + (1.0 - x[coordinate]) * np.sum(weights * right_slopes)
            )
            squared_slopes[coordinate] = (
                x[coordinate] * np.sum(weights * left_slopes**2)
                + (1.0 - x[coordinate]) * np.sum(weights * right_slopes**2)
            )
        contribution = np.outer(averaged_slopes, averaged_slopes)
        np.fill_diagonal(contribution, squared_slopes)
        row_contributions[row] = 0.5 * (contribution + contribution.T)

    estimates: dict[int, dict[str, object]] = {}
    for m in m_values:
        activity = row_contributions[:m].mean(axis=0)
        eigenvalues, eigenvectors, retained = eigenvectors_from_activity(activity)
        estimates[m] = {
            "activity": activity,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "retained_dimension": retained,
            "logical_evaluations": m * (1 + 2 * q * d),
        }
    return estimates


def basis_candidates(
    center: np.ndarray,
    vectors: np.ndarray,
    dimension: int,
    *,
    budget: int,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pairs = budget // 2
    full_radius_pairs = (pairs + 1) // 2
    radius_groups = [
        (full_radius_pairs, radius),
        (pairs - full_radius_pairs, radius / 2.0),
    ]
    directions: list[np.ndarray] = []
    radii: list[float] = []
    for group_size, group_radius in radius_groups:
        group_directions = [
            vectors[:, axis] for axis in range(min(dimension, group_size))
        ]
        if len(group_directions) < group_size:
            coefficients = rng.standard_normal(
                (group_size - len(group_directions), dimension)
            )
            combinations = normalize_rows(coefficients @ vectors[:, :dimension].T)
            group_directions.extend(combinations)
        directions.extend(group_directions)
        radii.extend([group_radius] * group_size)
    direction_array = np.asarray(directions)
    radius_array = np.asarray(radii)
    candidates = np.empty((budget, len(center)))
    candidates[0::2] = center + radius_array[:, None] * direction_array
    candidates[1::2] = center - radius_array[:, None] * direction_array
    return np.clip(candidates, 0.0, 1.0)


def random_candidates(
    center: np.ndarray,
    *,
    budget: int,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pairs = budget // 2
    full_radius_pairs = (pairs + 1) // 2
    directions = normalize_rows(rng.standard_normal((pairs, len(center))))
    radii = np.concatenate(
        [
            np.full(full_radius_pairs, radius),
            np.full(pairs - full_radius_pairs, radius / 2.0),
        ]
    )
    candidates = np.empty((budget, len(center)))
    candidates[0::2] = center + radii[:, None] * directions
    candidates[1::2] = center - radii[:, None] * directions
    return np.clip(candidates, 0.0, 1.0)


def select_and_verify(
    objective: MixtureObjective,
    candidates: np.ndarray,
    *,
    selection_seeds: list[int],
    verification_seeds: list[int],
    uniform_loss: float,
) -> dict[str, object]:
    selection_losses = np.asarray(
        [mean_objective(objective, candidate, selection_seeds) for candidate in candidates]
    )
    chosen = int(np.argmin(selection_losses))
    selected_u = candidates[chosen]
    verified_loss = mean_objective(objective, selected_u, verification_seeds)
    mixture = mixture_from_u(selected_u)
    return {
        "selected_candidate": chosen,
        "selection_loss": float(selection_losses[chosen]),
        "verified_loss": verified_loss,
        "uniform_verified_loss": uniform_loss,
        "verified_improvement": uniform_loss - verified_loss,
        **{
            f"weight_{name}": float(mixture[index])
            for index, name in enumerate(SOURCE_NAMES)
        },
    }


def summarize(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    value: str,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else groups
    for keys, subset in frame.groupby(grouper):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean, sd, low, high = t_interval(subset[value].to_numpy())
        row = {name: key for name, key in zip(groups, keys)}
        row.update(
            {
                "n_repeats": len(subset),
                f"mean_{prefix}": mean,
                f"sd_{prefix}": sd,
                f"ci95_low_{prefix}": low,
                f"ci95_high_{prefix}": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference_summary(values: pd.Series) -> dict[str, float | int]:
    """Summarize a paired difference whose index is the experimental repeat."""
    mean, sd, low, high = t_interval(values.to_numpy())
    return {
        "n_repeats": len(values),
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": low,
        "ci95_high": high,
    }


def make_plot(
    diagnostic_summary: pd.DataFrame,
    search_summary: pd.DataFrame,
    reference_evaluations: int,
    out: Path,
) -> None:
    colors = {
        "crn": "#4F81BD",
        "independent": "#F28E2B",
        "gas_crn": "#4F81BD",
        "gas_reference": "#17365D",
        "random": "#8C8C8C",
        "coordinate": "#59A14F",
    }
    evaluations = sorted(diagnostic_summary.logical_evaluations.unique().tolist())
    image = Image.new("RGB", (1900, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 35), "Data-mixture GAS direction-estimation budget curve", fill="#17365D", font=font(32, True))
    draw.text(
        (67, 82),
        "Same six-source loss-reweighting objective; only the nested outer-sample budget changes.",
        fill="#666666",
        font=font(17),
    )
    draw.text(
        (67, 112),
        f"The horizontal GAS-reference comparator uses a fixed {reference_evaluations}-call direction estimate.",
        fill="#666666",
        font=font(15),
    )

    panels = [
        (100, "Leading-direction cosine", 0.0, 1.0),
        (720, "Estimated 90% dimension", 0.0, 5.3),
        (1340, "Verified improvement over uniform", None, None),
    ]
    y0, width, height = 230, 470, 570
    panel_x_positions = {
        x0: [x0 + 45 + 95 * index for index in range(len(evaluations))]
        for x0, _, _, _ in panels
    }
    for x0, title, _, _ in panels:
        draw.text((x0, 165), title, fill="#17365D", font=font(20, True))
        draw.line((x0, y0, x0, y0 + height), fill="#333333", width=2)
        draw.line((x0, y0 + height, x0 + width, y0 + height), fill="#333333", width=2)

    # Direction cosine.
    x0 = panels[0][0]
    panel1_x = panel_x_positions[x0]
    for tick in range(6):
        yy = y0 + height - int(height * tick / 5)
        draw.text((x0 - 47, yy - 10), f"{tick / 5:.1f}", fill="#666666", font=font(13))
    for protocol in ["crn", "independent"]:
        subset = diagnostic_summary[diagnostic_summary.protocol == protocol].set_index("logical_evaluations")
        points: list[tuple[int, int]] = []
        for xx, evaluation in zip(panel1_x, evaluations):
            row = subset.loc[evaluation]
            mean = float(row.mean_top_direction_cosine)
            low = max(0.0, float(row.ci95_low_top_direction_cosine))
            high = min(1.0, float(row.ci95_high_top_direction_cosine))
            transform = lambda value: y0 + height - int(height * value)
            draw.line((xx, transform(high), xx, transform(low)), fill=colors[protocol], width=3)
            yy = transform(mean)
            draw.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill=colors[protocol])
            points.append((xx, yy))
        draw.line(points, fill=colors[protocol], width=4)

    # Retained dimension.
    x1 = panels[1][0]
    panel2_x = panel_x_positions[x1]
    for tick in range(6):
        yy = y0 + height - int(height * tick / 5)
        draw.text((x1 - 35, yy - 10), str(tick), fill="#666666", font=font(13))
    for protocol in ["crn", "independent"]:
        subset = diagnostic_summary[diagnostic_summary.protocol == protocol].set_index("logical_evaluations")
        points = []
        for xx, evaluation in zip(panel2_x, evaluations):
            mean = float(subset.loc[evaluation, "mean_retained_dimension"])
            yy = y0 + height - int(height * mean / 5.0)
            draw.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill=colors[protocol])
            points.append((xx, yy))
        draw.line(points, fill=colors[protocol], width=4)

    # Search outcome.
    x2 = panels[2][0]
    search_min = min(0.0, float(search_summary.ci95_low_verified_improvement.min()))
    search_max = max(0.01, float(search_summary.ci95_high_verified_improvement.max()) * 1.15)
    panel3_x = panel_x_positions[x2]
    for tick in range(6):
        value = search_min + (search_max - search_min) * tick / 5
        yy = y0 + height - int(height * tick / 5)
        draw.text((x2 - 67, yy - 10), f"{value:.3f}", fill="#666666", font=font(12))
    for method in ["gas_crn", "gas_reference", "random", "coordinate"]:
        subset = search_summary[search_summary.method == method].set_index("logical_evaluations")
        points = []
        for xx, evaluation in zip(panel3_x, evaluations):
            row = subset.loc[evaluation]
            mean = float(row.mean_verified_improvement)
            low = float(row.ci95_low_verified_improvement)
            high = float(row.ci95_high_verified_improvement)
            transform = lambda value: y0 + int(height * (search_max - value) / (search_max - search_min))
            draw.line((xx, transform(high), xx, transform(low)), fill=colors[method], width=2)
            yy = transform(mean)
            draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=colors[method])
            points.append((xx, yy))
        draw.line(points, fill=colors[method], width=4)

    for panel_xs in [panel1_x, panel2_x, panel3_x]:
        for xx, evaluation in zip(panel_xs, evaluations):
            draw.text((xx - 22, y0 + height + 15), str(evaluation), fill="#333333", font=font(13))
    draw.text((700, y0 + height + 58), "logical train-and-evaluate calls used for one GAS estimate", fill="#666666", font=font(15))

    legend_y = 930
    legend = [
        ("crn", "CRN direction estimate"),
        ("independent", "Independent direction estimate"),
        ("gas_reference", "High-budget GAS reference"),
        ("random", "Random search"),
        ("coordinate", "Coordinate search"),
    ]
    for index, (key, label) in enumerate(legend):
        xx = 80 + index * 360
        draw.line((xx, legend_y, xx + 42, legend_y), fill=colors[key], width=5)
        draw.text((xx + 52, legend_y - 12), label, fill="#222222", font=font(13))
    image.save(out)


def run(
    out_dir: Path,
    *,
    training_steps: int = 120,
    m_values: list[int] | None = None,
    q: int = 2,
    repeats: int = 6,
    reference_m: int = 24,
    reference_q: int = 3,
    candidate_budget: int = 20,
    selection_seed_count: int = 2,
    verification_seed_count: int = 8,
    search_radius: float = 0.32,
) -> None:
    if m_values is None:
        m_values = [2, 4, 8, 16, 24]
    m_values = sorted(set(m_values))
    if any(value <= 0 for value in m_values):
        raise ValueError("All m values must be positive")
    if candidate_budget <= 0 or candidate_budget % 2:
        raise ValueError("Candidate budget must be a positive even integer")

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    objective = MixtureObjective(generate_data(), steps=training_steps)
    reference_activity, reference_eigenvalues, reference_dimension, reference_evaluations = activity_from_objective(
        objective,
        m=reference_m,
        q=reference_q,
        seed=2026101510,
        crn=True,
    )
    reference_unique_trainings = len(objective.cache)
    _, reference_vectors, _ = eigenvectors_from_activity(reference_activity)

    diagnostics: list[dict[str, object]] = []
    estimates_by_repeat: dict[int, dict[int, dict[str, object]]] = {}
    for repeat in range(repeats):
        repeat_seed = 2026103000 + 100 * repeat
        for protocol, crn in [("crn", True), ("independent", False)]:
            estimates = nested_activity_estimates(
                objective,
                m_values=m_values,
                q=q,
                seed=repeat_seed,
                crn=crn,
            )
            if crn:
                estimates_by_repeat[repeat] = estimates
            for m, estimate in estimates.items():
                overlap, angle, cosine = subspace_metrics(
                    reference_vectors,
                    estimate["eigenvectors"],
                    reference_dimension,
                )
                normalized_reference = reference_activity / max(
                    float(np.trace(reference_activity)), 1e-12
                )
                normalized_estimate = estimate["activity"] / max(
                    float(np.trace(estimate["activity"])), 1e-12
                )
                matrix_error = float(
                    np.linalg.norm(normalized_estimate - normalized_reference, ord="fro")
                )
                diagnostics.append(
                    {
                        "repeat": repeat,
                        "protocol": protocol,
                        "m": m,
                        "q": q,
                        "logical_evaluations": estimate["logical_evaluations"],
                        "retained_dimension": estimate["retained_dimension"],
                        "reference_dimension": reference_dimension,
                        "subspace_overlap": overlap,
                        "max_principal_angle": angle,
                        "top_direction_cosine": cosine,
                        "trace_normalized_frobenius_error": matrix_error,
                    }
                )

    direction_curve_unique_trainings = len(objective.cache) - reference_unique_trainings

    diagnostic_df = pd.DataFrame(diagnostics)
    diagnostic_summaries = []
    for metric in [
        "top_direction_cosine",
        "retained_dimension",
        "subspace_overlap",
        "max_principal_angle",
        "trace_normalized_frobenius_error",
    ]:
        metric_summary = summarize(
            diagnostic_df,
            groups=["protocol", "m", "q", "logical_evaluations"],
            value=metric,
            prefix=metric,
        )
        diagnostic_summaries.append(metric_summary)
    diagnostic_summary = diagnostic_summaries[0]
    for extra in diagnostic_summaries[1:]:
        diagnostic_summary = diagnostic_summary.merge(
            extra,
            on=["protocol", "m", "q", "logical_evaluations", "n_repeats"],
        )

    minimum_m = min(m_values)
    minimum_evaluations = minimum_m * (1 + 2 * q * (len(SOURCE_NAMES) - 1))
    direction_budget_contrasts: list[dict[str, object]] = []
    crn_diagnostics = diagnostic_df[diagnostic_df.protocol == "crn"]
    for metric, positive_means in [
        ("top_direction_cosine", "higher-budget estimate is closer to the reference"),
        (
            "trace_normalized_frobenius_error",
            "higher-budget estimate is closer to the reference",
        ),
    ]:
        pivot = crn_diagnostics.pivot(index="repeat", columns="m", values=metric)
        for m in m_values[1:]:
            differences = (
                pivot[m] - pivot[minimum_m]
                if metric == "top_direction_cosine"
                else pivot[minimum_m] - pivot[m]
            )
            direction_budget_contrasts.append(
                {
                    "contrast_family": "crn_budget_vs_minimum",
                    "metric": metric,
                    "m_low": minimum_m,
                    "m_high": m,
                    "logical_evaluations_low": minimum_evaluations,
                    "logical_evaluations_high": m
                    * (1 + 2 * q * (len(SOURCE_NAMES) - 1)),
                    "positive_means": positive_means,
                    **paired_difference_summary(differences),
                }
            )
    for m in m_values:
        at_budget = diagnostic_df[diagnostic_df.m == m]
        for metric, minuend, subtrahend, positive_means in [
            (
                "top_direction_cosine",
                "crn",
                "independent",
                "CRN estimate is closer to the reference",
            ),
            (
                "trace_normalized_frobenius_error",
                "independent",
                "crn",
                "CRN estimate is closer to the reference",
            ),
        ]:
            pivot = at_budget.pivot(index="repeat", columns="protocol", values=metric)
            differences = pivot[minuend] - pivot[subtrahend]
            evaluations = m * (1 + 2 * q * (len(SOURCE_NAMES) - 1))
            direction_budget_contrasts.append(
                {
                    "contrast_family": "crn_vs_independent",
                    "metric": metric,
                    "m_low": m,
                    "m_high": m,
                    "logical_evaluations_low": evaluations,
                    "logical_evaluations_high": evaluations,
                    "positive_means": positive_means,
                    **paired_difference_summary(differences),
                }
            )
    direction_budget_contrast_df = pd.DataFrame(direction_budget_contrasts)

    center = np.full(len(SOURCE_NAMES) - 1, 0.5)
    search_rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        selection_seeds = [
            2026130000 + 1000 * repeat + index for index in range(selection_seed_count)
        ]
        verification_seeds = [
            2026140000 + 1000 * repeat + index for index in range(verification_seed_count)
        ]
        uniform_loss = mean_objective(objective, center, verification_seeds)
        baseline_candidates = {
            "gas_reference": basis_candidates(
                center,
                reference_vectors,
                reference_dimension,
                budget=candidate_budget,
                radius=search_radius,
                rng=np.random.default_rng(2026104000 + 100 * repeat + 1),
            ),
            "random": random_candidates(
                center,
                budget=candidate_budget,
                radius=search_radius,
                rng=np.random.default_rng(2026104000 + 100 * repeat + 2),
            ),
            "coordinate": basis_candidates(
                center,
                np.eye(len(center)),
                len(center),
                budget=candidate_budget,
                radius=search_radius,
                rng=np.random.default_rng(2026104000 + 100 * repeat + 3),
            ),
        }
        baseline_results = {
            method: select_and_verify(
                objective,
                candidates,
                selection_seeds=selection_seeds,
                verification_seeds=verification_seeds,
                uniform_loss=uniform_loss,
            )
            for method, candidates in baseline_candidates.items()
        }

        for m in m_values:
            estimate = estimates_by_repeat[repeat][m]
            gas_candidates = basis_candidates(
                center,
                estimate["eigenvectors"],
                int(estimate["retained_dimension"]),
                budget=candidate_budget,
                radius=search_radius,
                rng=np.random.default_rng(2026105000 + 1000 * repeat),
            )
            gas_result = select_and_verify(
                objective,
                gas_candidates,
                selection_seeds=selection_seeds,
                verification_seeds=verification_seeds,
                uniform_loss=uniform_loss,
            )
            logical_evaluations = int(estimate["logical_evaluations"])
            search_rows.append(
                {
                    "repeat": repeat,
                    "method": "gas_crn",
                    "m": m,
                    "q": q,
                    "logical_evaluations": logical_evaluations,
                    **gas_result,
                }
            )
            for method, result in baseline_results.items():
                search_rows.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "m": m,
                        "q": q,
                        "logical_evaluations": logical_evaluations,
                        **result,
                    }
                )

    search_df = pd.DataFrame(search_rows)
    search_summary = summarize(
        search_df,
        groups=["method", "m", "q", "logical_evaluations"],
        value="verified_improvement",
        prefix="verified_improvement",
    )
    contrast_rows: list[dict[str, object]] = []
    for m, subset in search_df.groupby("m"):
        pivot = subset.pivot(index="repeat", columns="method", values="verified_improvement")
        for minuend, subtrahend, label in [
            ("gas_crn", "random", "gas_crn_minus_random"),
            ("gas_crn", "coordinate", "gas_crn_minus_coordinate"),
            ("gas_reference", "gas_crn", "reference_minus_gas_crn"),
        ]:
            differences = pivot[minuend] - pivot[subtrahend]
            mean, sd, low, high = t_interval(differences.to_numpy())
            contrast_rows.append(
                {
                    "m": int(m),
                    "q": q,
                    "logical_evaluations": int(m) * (1 + 2 * q * (len(SOURCE_NAMES) - 1)),
                    "contrast": label,
                    "n_repeats": len(differences),
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    contrast_df = pd.DataFrame(contrast_rows)

    gas_curve = search_df[search_df.method == "gas_crn"].pivot(
        index="repeat", columns="m", values="verified_improvement"
    )
    search_budget_contrast_rows: list[dict[str, object]] = []
    for m in m_values[1:]:
        differences = gas_curve[m] - gas_curve[minimum_m]
        search_budget_contrast_rows.append(
            {
                "m_low": minimum_m,
                "m_high": m,
                "logical_evaluations_low": minimum_evaluations,
                "logical_evaluations_high": m
                * (1 + 2 * q * (len(SOURCE_NAMES) - 1)),
                "positive_means": "higher-budget CRN-GAS search improves more",
                **paired_difference_summary(differences),
            }
        )
    search_budget_contrast_df = pd.DataFrame(search_budget_contrast_rows)

    expected_diagnostic_rows = repeats * 2 * len(m_values)
    expected_search_rows = repeats * 4 * len(m_values)
    if len(diagnostic_df) != expected_diagnostic_rows or diagnostic_df.isna().any().any():
        raise RuntimeError("Direction diagnostics failed row-count or missing-value checks")
    if len(search_df) != expected_search_rows or search_df.isna().any().any():
        raise RuntimeError("Search results failed row-count or missing-value checks")
    weight_columns = [column for column in search_df if column.startswith("weight_")]
    if not np.allclose(search_df[weight_columns].sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("At least one selected mixture does not lie on the simplex")
    if not np.allclose(
        search_df.uniform_verified_loss - search_df.verified_loss,
        search_df.verified_improvement,
        atol=1e-12,
    ):
        raise RuntimeError("Verified-improvement accounting is inconsistent")
    baseline_variation = (
        search_df[search_df.method.isin(["gas_reference", "random", "coordinate"])]
        .groupby(["repeat", "method"])
        .verified_improvement.nunique()
    )
    if int(baseline_variation.max()) != 1:
        raise RuntimeError("A fixed search baseline changed across GAS budgets")

    diagnostic_df.to_csv(out_dir / "direction_diagnostics_by_repeat.csv", index=False)
    diagnostic_summary.to_csv(out_dir / "direction_diagnostics_summary.csv", index=False)
    direction_budget_contrast_df.to_csv(
        out_dir / "direction_budget_contrasts.csv", index=False
    )
    search_df.to_csv(out_dir / "budget_search_by_repeat.csv", index=False)
    search_summary.to_csv(out_dir / "budget_search_summary.csv", index=False)
    contrast_df.to_csv(out_dir / "budget_search_contrasts.csv", index=False)
    search_budget_contrast_df.to_csv(
        out_dir / "budget_search_budget_contrasts.csv", index=False
    )
    pd.DataFrame(
        {
            "eigenvalue_index": np.arange(1, len(reference_eigenvalues) + 1),
            "eigenvalue": reference_eigenvalues,
            "energy_fraction": reference_eigenvalues / reference_eigenvalues.sum(),
            "cumulative_energy": np.cumsum(reference_eigenvalues) / reference_eigenvalues.sum(),
        }
    ).to_csv(out_dir / "reference_activity_spectrum.csv", index=False)
    make_plot(
        diagnostic_summary,
        search_summary,
        reference_evaluations,
        out_dir / "data_mixture_budget_curve.png",
    )

    metadata = {
        "training_steps_per_objective_evaluation": training_steps,
        "m_values": m_values,
        "q": q,
        "logical_evaluation_budgets": [
            value * (1 + 2 * q * (len(SOURCE_NAMES) - 1)) for value in m_values
        ],
        "nested_design": True,
        "protocols": ["crn", "independent"],
        "repeats": repeats,
        "reference_m": reference_m,
        "reference_q": reference_q,
        "reference_evaluations": reference_evaluations,
        "reference_unique_trainings": reference_unique_trainings,
        "reference_retained_dimension": reference_dimension,
        "candidate_budget_per_method": candidate_budget,
        "selection_training_seeds_per_candidate": selection_seed_count,
        "verification_training_seeds_per_selected_mixture": verification_seed_count,
        "search_radius_in_log_ratio_coordinates": search_radius,
        "candidate_radius_schedule": (
            f"Every method uses {(candidate_budget // 2 + 1) // 2} paired "
            "directions at the full radius and "
            f"{candidate_budget // 2 - (candidate_budget // 2 + 1) // 2} paired "
            "directions at half radius."
        ),
        "budget_interpretation": (
            "Logical costs describe a standalone GAS estimate at each prefix. "
            "Nested execution reuses every smaller prefix inside the largest design."
        ),
        "search_interpretation": (
            "Search curves condition on the estimated directions and hold the "
            f"{candidate_budget}-candidate search budget fixed; GAS direction-estimation calls are "
            "reported separately and are not included in that candidate budget."
        ),
        "adaptive_search_dimension": (
            "Each CRN-GAS search uses its estimate's own 90%-energy dimension. "
            "The search curve therefore evaluates the complete estimated-basis "
            "procedure, not orientation alone."
        ),
        "independent_protocol_interpretation": (
            "This is a deliberately naive independent-seed finite-difference "
            "control. Near-boundary differences can divide independent training "
            "noise by a small coordinate displacement, producing heavy-tailed errors."
        ),
        "reference_interpretation": (
            "A fixed higher-budget CRN estimate used as a pathwise benchmark, not ground truth."
        ),
        "direction_curve_statistical_unit": "independent nested-design repeat conditional on one fixed synthetic data problem",
        "search_statistical_unit": "independent direction-estimation and candidate-search repeat conditional on one fixed synthetic data problem",
        "reference_design_seed": 2026101510,
        "direction_design_seed_formula": "2026103000 + 100 * repeat",
        "search_selection_seed_formula": "2026130000 + 1000 * repeat + index",
        "search_verification_seed_formula": "2026140000 + 1000 * repeat + index",
        "gas_candidate_seed_formula": "2026105000 + 1000 * repeat (shared across budgets)",
        "runtime_seconds": time.perf_counter() - started,
        "direction_curve_incremental_unique_trainings": direction_curve_unique_trainings,
        "search_incremental_unique_trainings": (
            len(objective.cache)
            - reference_unique_trainings
            - direction_curve_unique_trainings
        ),
        "objective_cache_entries": len(objective.cache),
    }
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nDirection diagnostics summary:")
    print(
        diagnostic_summary[
            [
                "protocol",
                "logical_evaluations",
                "mean_top_direction_cosine",
                "ci95_low_top_direction_cosine",
                "ci95_high_top_direction_cosine",
                "mean_retained_dimension",
                "mean_trace_normalized_frobenius_error",
            ]
        ].to_string(index=False)
    )
    print("\nSearch contrasts:")
    print(contrast_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--training-steps", type=int, default=120)
    parser.add_argument("--m-values", type=int, nargs="+", default=[2, 4, 8, 16, 24])
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--reference-m", type=int, default=24)
    parser.add_argument("--reference-q", type=int, default=3)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--selection-seeds", type=int, default=2)
    parser.add_argument("--verification-seeds", type=int, default=8)
    parser.add_argument("--search-radius", type=float, default=0.32)
    args = parser.parse_args()
    run(
        args.out,
        training_steps=args.training_steps,
        m_values=args.m_values,
        q=args.q,
        repeats=args.repeats,
        reference_m=args.reference_m,
        reference_q=args.reference_q,
        candidate_budget=args.candidate_budget,
        selection_seed_count=args.selection_seeds,
        verification_seed_count=args.verification_seeds,
        search_radius=args.search_radius,
    )
