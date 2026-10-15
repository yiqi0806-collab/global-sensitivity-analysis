"""Pilot study for GAS-guided data-mixture optimization.

Six synthetic data sources differ in domain coverage and label quality.  A
small ReLU MLP is retrained for every mixture, so the black-box objective is the
clean validation loss after training.  Mixture weights live on a simplex and
are represented by orthonormal log-ratio coordinates.

The study asks three questions:

1. Can low-budget CRN-GAS recover a stable activity subspace for the stochastic
   train-and-evaluate objective?
2. Does using independent training randomness for paired evaluations create a
   noisy, high-dimensional activity estimate?
3. Do CRN-GAS directions find better data mixtures than random or coordinate
   directions at the same candidate budget?
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

from neural_network_gas_demo import font
from neural_network_gas_direction_intervention import normalize_rows
from neural_network_gas_mc_dropout import subspace_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "data_mixture_pilot"
SOURCE_NAMES = [
    "clean_left",
    "clean_right",
    "clean_center",
    "noisy_left",
    "biased_right",
    "corrupted_center",
]
METHODS = ["gas_reference", "gas_crn", "gas_independent", "random", "coordinate"]
METHOD_LABELS = {
    "gas_reference": "GAS reference",
    "gas_crn": "CRN-GAS",
    "gas_independent": "Independent GAS",
    "random": "Random directions",
    "coordinate": "Coordinate directions",
}


@dataclass
class MixtureData:
    x_sources: list[np.ndarray]
    y_sources: list[np.ndarray]
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def true_response(x: np.ndarray) -> np.ndarray:
    return (
        1.5 * np.sin(x[:, 0])
        + 0.65 * x[:, 1] ** 2
        - 0.9 * x[:, 2] * x[:, 3]
        + 0.45 * x[:, 4]
        + 0.25 * np.cos(1.5 * x[:, 5])
    )


def generate_data(seed: int = 2026101501, n_per_source: int = 160) -> MixtureData:
    rng = np.random.default_rng(seed)
    shifts = [-1.35, 1.35, 0.0, -1.35, 1.35, 0.0]
    x_sources: list[np.ndarray] = []
    y_sources: list[np.ndarray] = []
    for source, shift in enumerate(shifts):
        x = rng.normal(0.0, 1.0, size=(n_per_source, 6))
        x[:, 0] += shift
        clean = true_response(x)
        if source <= 2:
            y = clean + rng.normal(0.0, 0.12, size=n_per_source)
        elif source == 3:
            y = clean + 0.55 + rng.normal(0.0, 0.85, size=n_per_source)
        elif source == 4:
            y = clean - 0.70 + 0.35 * x[:, 1] + rng.normal(0.0, 0.75, size=n_per_source)
        else:
            y = rng.permutation(clean) + rng.normal(0.0, 1.10, size=n_per_source)
        x_sources.append(x)
        y_sources.append(y)

    validation_blocks = []
    for shift in (-1.35, 0.0, 1.35):
        block = rng.normal(0.0, 1.0, size=(300, 6))
        block[:, 0] += shift
        validation_blocks.append(block)
    x_validation = np.vstack(validation_blocks)
    y_validation = true_response(x_validation)

    pooled_x = np.vstack(x_sources)
    x_mean = pooled_x.mean(axis=0)
    x_std = pooled_x.std(axis=0, ddof=1)
    y_mean = float(y_validation.mean())
    y_std = float(y_validation.std(ddof=1))
    return MixtureData(
        x_sources=[(x - x_mean) / x_std for x in x_sources],
        y_sources=[(y - y_mean) / y_std for y in y_sources],
        x_validation=(x_validation - x_mean) / x_std,
        y_validation=(y_validation - y_mean) / y_std,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


def helmert_basis(k: int) -> np.ndarray:
    basis = np.zeros((k, k - 1))
    for column in range(k - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def mixture_from_u(u: np.ndarray, *, scale: float = 2.25) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    basis = helmert_basis(len(SOURCE_NAMES))
    logits = scale * (basis @ (2.0 * u - 1.0))
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


class MixtureObjective:
    def __init__(
        self,
        data: MixtureData,
        *,
        steps: int = 120,
        width: int = 14,
        batch_per_source: int = 8,
    ) -> None:
        self.data = data
        self.steps = steps
        self.width = width
        self.batch_per_source = batch_per_source
        self.cache: dict[tuple[bytes, int], float] = {}

    def __call__(self, u: np.ndarray, training_seed: int) -> float:
        clipped = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        key = (np.round(clipped, 12).tobytes(), int(training_seed))
        if key not in self.cache:
            weights = mixture_from_u(clipped)
            self.cache[key] = self._train_and_validate(weights, training_seed)
        return self.cache[key]

    def _train_and_validate(self, mixture: np.ndarray, seed: int) -> float:
        rng = np.random.default_rng(seed)
        d = self.data.x_sources[0].shape[1]
        width = self.width
        w1 = rng.normal(0.0, math.sqrt(2.0 / d), size=(d, width))
        b1 = np.zeros(width)
        w2 = rng.normal(0.0, math.sqrt(2.0 / width), size=(width, 1))
        b2 = np.zeros(1)
        params = [w1, b1, w2, b2]
        adam_m = [np.zeros_like(param) for param in params]
        adam_v = [np.zeros_like(param) for param in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, self.steps + 1):
            x_blocks: list[np.ndarray] = []
            y_blocks: list[np.ndarray] = []
            sample_weights: list[np.ndarray] = []
            for source, (x_source, y_source) in enumerate(
                zip(self.data.x_sources, self.data.y_sources)
            ):
                indices = rng.integers(0, len(x_source), size=self.batch_per_source)
                x_blocks.append(x_source[indices])
                y_blocks.append(y_source[indices])
                sample_weights.append(
                    np.full(self.batch_per_source, mixture[source] / self.batch_per_source)
                )
            x_batch = np.vstack(x_blocks)
            y_batch = np.concatenate(y_blocks)
            row_weights = np.concatenate(sample_weights)

            z1 = x_batch @ w1 + b1
            a1 = np.maximum(0.0, z1)
            prediction = (a1 @ w2 + b2)[:, 0]
            grad_prediction = 2.0 * row_weights * (prediction - y_batch)
            grad_w2 = a1.T @ grad_prediction[:, None] + 2e-4 * w2
            grad_b2 = np.asarray([grad_prediction.sum()])
            grad_a1 = grad_prediction[:, None] @ w2.T
            grad_z1 = grad_a1 * (z1 > 0.0)
            grad_w1 = x_batch.T @ grad_z1 + 2e-4 * w1
            grad_b1 = grad_z1.sum(axis=0)
            gradients = [grad_w1, grad_b1, grad_w2, grad_b2]

            learning_rate = 0.012 if step <= int(0.7 * self.steps) else 0.004
            for index, (parameter, gradient) in enumerate(zip(params, gradients)):
                adam_m[index] = beta1 * adam_m[index] + (1.0 - beta1) * gradient
                adam_v[index] = beta2 * adam_v[index] + (1.0 - beta2) * gradient**2
                m_hat = adam_m[index] / (1.0 - beta1**step)
                v_hat = adam_v[index] / (1.0 - beta2**step)
                parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        hidden = np.maximum(0.0, self.data.x_validation @ w1 + b1)
        prediction = (hidden @ w2 + b2)[:, 0]
        return float(np.mean((prediction - self.data.y_validation) ** 2))


def activity_from_objective(
    objective: MixtureObjective,
    *,
    m: int,
    q: int,
    seed: int,
    crn: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    d = len(SOURCE_NAMES) - 1
    points = rng.random((m, d))
    base_seeds = rng.integers(1, 2**31 - 1, size=m)
    base_values = np.asarray(
        [objective(point, int(run_seed)) for point, run_seed in zip(points, base_seeds)]
    )
    nodes_raw, weights_raw = np.polynomial.legendre.leggauss(q)
    nodes = 0.5 * (nodes_raw + 1.0)
    quadrature_weights = 0.5 * weights_raw
    averaged_slopes = np.zeros((m, d))
    squared_slopes = np.zeros((m, d))

    for coordinate in range(d):
        for row in range(m):
            x = points[row]
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
            averaged_slopes[row, coordinate] = (
                x[coordinate] * np.sum(quadrature_weights * left_slopes)
                + (1.0 - x[coordinate]) * np.sum(quadrature_weights * right_slopes)
            )
            squared_slopes[row, coordinate] = (
                x[coordinate] * np.sum(quadrature_weights * left_slopes**2)
                + (1.0 - x[coordinate]) * np.sum(quadrature_weights * right_slopes**2)
            )

    activity = averaged_slopes.T @ averaged_slopes / m
    np.fill_diagonal(activity, squared_slopes.mean(axis=0))
    activity = 0.5 * (activity + activity.T)
    eigenvalues, eigenvectors = np.linalg.eigh(activity)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    energy = np.cumsum(eigenvalues) / max(float(eigenvalues.sum()), 1e-12)
    retained = int(np.searchsorted(energy, 0.90) + 1)
    evaluations = m * (1 + 2 * q * d)
    return activity, eigenvalues, retained, evaluations


def eigenvectors_from_activity(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    eigenvalues, eigenvectors = np.linalg.eigh(activity)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    energy = np.cumsum(eigenvalues) / max(float(eigenvalues.sum()), 1e-12)
    retained = int(np.searchsorted(energy, 0.90) + 1)
    return eigenvalues, eigenvectors, retained


def paired_u_candidates(center: np.ndarray, directions: np.ndarray, radius: float) -> np.ndarray:
    candidates = np.empty((2 * len(directions), len(center)))
    candidates[0::2] = center + radius * directions
    candidates[1::2] = center - radius * directions
    return np.clip(candidates, 0.0, 1.0)


def candidate_sets(
    *,
    center: np.ndarray,
    budget: int,
    radius: float,
    reference_vectors: np.ndarray,
    reference_dimension: int,
    crn_vectors: np.ndarray,
    crn_dimension: int,
    independent_vectors: np.ndarray,
    independent_dimension: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    pairs = budget // 2

    def from_basis(vectors: np.ndarray, dimension: int) -> np.ndarray:
        directions: list[np.ndarray] = []
        radii: list[float] = []
        # Test every learned activity axis directly before spending candidates
        # on combinations.  This makes the comparison analogous to coordinate
        # search, which explicitly tests its own basis axes.
        for scale_factor in (1.0, 0.5):
            for axis in range(dimension):
                if len(directions) == pairs:
                    break
                directions.append(vectors[:, axis])
                radii.append(radius * scale_factor)
        if len(directions) < pairs:
            coefficients = rng.standard_normal((pairs - len(directions), dimension))
            combinations = normalize_rows(coefficients @ vectors[:, :dimension].T)
            directions.extend(combinations)
            radii.extend([radius] * len(combinations))
        direction_array = np.asarray(directions)
        radius_array = np.asarray(radii)
        candidates = np.empty((2 * pairs, len(center)))
        candidates[0::2] = center + radius_array[:, None] * direction_array
        candidates[1::2] = center - radius_array[:, None] * direction_array
        return np.clip(candidates, 0.0, 1.0)

    random_directions = normalize_rows(rng.standard_normal((pairs, len(center))))
    coordinate_directions = np.vstack(
        [np.eye(len(center)), np.eye(len(center))]
    )[:pairs]
    coordinate_radii = np.repeat([radius, radius / 2.0], len(center))[:pairs]
    coordinate_candidates = np.empty((2 * pairs, len(center)))
    coordinate_candidates[0::2] = center + coordinate_radii[:, None] * coordinate_directions
    coordinate_candidates[1::2] = center - coordinate_radii[:, None] * coordinate_directions
    coordinate_candidates = np.clip(coordinate_candidates, 0.0, 1.0)
    return {
        "gas_reference": from_basis(reference_vectors, reference_dimension),
        "gas_crn": from_basis(crn_vectors, crn_dimension),
        "gas_independent": from_basis(independent_vectors, independent_dimension),
        "random": paired_u_candidates(center, random_directions, radius),
        "coordinate": coordinate_candidates,
    }


def mean_objective(
    objective: MixtureObjective, u: np.ndarray, seeds: list[int]
) -> float:
    return float(np.mean([objective(u, seed) for seed in seeds]))


def t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(len(values), 1.96)
    half_width = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sd, mean - half_width, mean + half_width


def make_plot(
    reference_eigenvalues: np.ndarray,
    diagnostics: pd.DataFrame,
    search_summary: pd.DataFrame,
    out: Path,
) -> None:
    colors = {
        "gas_reference": "#17365D",
        "gas_crn": "#4F81BD",
        "gas_independent": "#F28E2B",
        "random": "#8C8C8C",
        "coordinate": "#59A14F",
    }
    image = Image.new("RGB", (1900, 1040), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 35), "GAS-guided data-mixture optimization pilot", fill="#17365D", font=font(34, True))
    draw.text(
        (67, 85),
        "Six data sources; every objective evaluation retrains a small ReLU network.",
        fill="#666666",
        font=font(17),
    )

    # Reference eigenvalue energy.
    x0, y0, width, height = 100, 240, 470, 560
    draw.text((x0, 170), "Reference activity spectrum", fill="#17365D", font=font(21, True))
    draw.line((x0, y0, x0, y0 + height), fill="#333333", width=2)
    draw.line((x0, y0 + height, x0 + width, y0 + height), fill="#333333", width=2)
    fractions = reference_eigenvalues / reference_eigenvalues.sum()
    bar_width = 60
    for index, value in enumerate(fractions):
        xx = x0 + 35 + index * 85
        bar_height = int(height * float(value))
        draw.rectangle((xx, y0 + height - bar_height, xx + bar_width, y0 + height), fill="#4F81BD")
        draw.text((xx + 20, y0 + height + 12), str(index + 1), fill="#333333", font=font(15))
        draw.text((xx + 4, y0 + height - bar_height - 28), f"{value:.2f}", fill="#333333", font=font(13))

    # Leading-direction recovery.  With a four-dimensional reference subspace
    # inside a five-dimensional parameter space, random subspaces already have
    # substantial overlap, so the leading-direction cosine is more diagnostic.
    x1, y1, w1, h1 = 715, 240, 470, 560
    draw.text((x1, 170), "Leading-direction recovery", fill="#17365D", font=font(21, True))
    draw.line((x1, y1, x1, y1 + h1), fill="#333333", width=2)
    draw.line((x1, y1 + h1, x1 + w1, y1 + h1), fill="#333333", width=2)
    for tick in range(6):
        yy = y1 + h1 - int(h1 * tick / 5)
        draw.text((x1 - 48, yy - 10), f"{tick / 5:.1f}", fill="#666666", font=font(13))
    for index, protocol in enumerate(["crn", "independent"]):
        values = diagnostics[diagnostics.protocol == protocol].top_direction_cosine.to_numpy()
        mean, _, low, high = t_interval(values)
        low = max(0.0, low)
        high = min(1.0, high)
        xx = x1 + 145 + index * 210
        transform = lambda value: y1 + h1 - int(h1 * value)
        draw.line((xx, transform(high), xx, transform(low)), fill=colors["gas_crn" if protocol == "crn" else "gas_independent"], width=4)
        yy = transform(mean)
        draw.ellipse((xx - 9, yy - 9, xx + 9, yy + 9), fill=colors["gas_crn" if protocol == "crn" else "gas_independent"])
        label = "CRN" if protocol == "crn" else "Independent"
        draw.text((xx - 50, y1 + h1 + 20), label, fill="#333333", font=font(15))

    # Search improvement.
    x2, y2, w2, h2 = 1330, 240, 470, 560
    draw.text((x2, 170), "Verified improvement over uniform", fill="#17365D", font=font(21, True))
    draw.line((x2, y2, x2, y2 + h2), fill="#333333", width=2)
    draw.line((x2, y2 + h2, x2 + w2, y2 + h2), fill="#333333", width=2)
    y_min = min(0.0, float(search_summary.ci95_low.min()))
    y_max = max(0.05, float(search_summary.ci95_high.max()) * 1.15)
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        yy = y2 + h2 - int(h2 * tick / 5)
        draw.line((x2 - 6, yy, x2, yy), fill="#333333", width=2)
        draw.text((x2 - 70, yy - 10), f"{value:.3f}", fill="#666666", font=font(12))
    for index, row in enumerate(search_summary.itertuples(index=False)):
        xx = x2 + 45 + index * 88
        transform = lambda value: y2 + int(h2 * (y_max - value) / (y_max - y_min))
        draw.line((xx, transform(row.ci95_high), xx, transform(row.ci95_low)), fill=colors[row.method], width=3)
        yy = transform(row.mean_improvement)
        draw.ellipse((xx - 7, yy - 7, xx + 7, yy + 7), fill=colors[row.method])
        draw.text((xx - 27, y2 + h2 + 20), str(index + 1), fill="#333333", font=font(14))
    legend_y = 910
    for index, method in enumerate(METHODS):
        xx = 110 + index * 350
        draw.line((xx, legend_y, xx + 40, legend_y), fill=colors[method], width=5)
        draw.text((xx + 52, legend_y - 12), f"{index + 1}: {METHOD_LABELS[method]}", fill="#222222", font=font(14))
    image.save(out)


def run(
    out_dir: Path,
    *,
    training_steps: int = 120,
    reference_m: int = 24,
    reference_q: int = 3,
    low_m: int = 8,
    low_q: int = 2,
    sensitivity_repeats: int = 5,
    search_repeats: int = 6,
    candidate_budget: int = 20,
    selection_seeds: int = 2,
    verification_seeds: int = 12,
    search_radius: float = 0.32,
) -> None:
    if candidate_budget <= 0 or candidate_budget % 2:
        raise ValueError("Candidate budget must be a positive even integer")
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate_data()
    objective = MixtureObjective(data, steps=training_steps)

    reference_activity, reference_eigenvalues, reference_dimension, reference_evals = activity_from_objective(
        objective,
        m=reference_m,
        q=reference_q,
        seed=2026101510,
        crn=True,
    )
    _, reference_vectors, _ = eigenvectors_from_activity(reference_activity)

    diagnostics: list[dict[str, object]] = []
    first_estimate: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for repeat in range(sensitivity_repeats):
        for protocol, crn in [("crn", True), ("independent", False)]:
            activity, eigenvalues, retained, evaluations = activity_from_objective(
                objective,
                m=low_m,
                q=low_q,
                seed=2026101600 + 100 * repeat,
                crn=crn,
            )
            _, vectors, _ = eigenvectors_from_activity(activity)
            overlap, angle, cosine = subspace_metrics(
                reference_vectors, vectors, reference_dimension
            )
            diagnostics.append(
                {
                    "repeat": repeat,
                    "protocol": protocol,
                    "retained_dimension": retained,
                    "reference_dimension": reference_dimension,
                    "subspace_overlap": overlap,
                    "max_principal_angle": angle,
                    "top_direction_cosine": cosine,
                    "direction_evaluations": evaluations,
                }
            )
            if repeat == 0:
                first_estimate[protocol] = (eigenvalues, vectors, retained)

    center = np.full(len(SOURCE_NAMES) - 1, 0.5)
    search_rows: list[dict[str, object]] = []
    for repeat in range(search_repeats):
        rng = np.random.default_rng(2026102000 + repeat)
        pools = candidate_sets(
            center=center,
            budget=candidate_budget,
            radius=search_radius,
            reference_vectors=reference_vectors,
            reference_dimension=reference_dimension,
            crn_vectors=first_estimate["crn"][1],
            crn_dimension=first_estimate["crn"][2],
            independent_vectors=first_estimate["independent"][1],
            independent_dimension=first_estimate["independent"][2],
            rng=rng,
        )
        select = [2026110000 + 1000 * repeat + index for index in range(selection_seeds)]
        verify = [2026120000 + 1000 * repeat + index for index in range(verification_seeds)]
        uniform_loss = mean_objective(objective, center, verify)
        for method in METHODS:
            candidates = pools[method]
            selection_losses = np.asarray(
                [mean_objective(objective, candidate, select) for candidate in candidates]
            )
            chosen = int(np.argmin(selection_losses))
            selected_u = candidates[chosen]
            verified_loss = mean_objective(objective, selected_u, verify)
            mixture = mixture_from_u(selected_u)
            search_rows.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "selected_candidate": chosen,
                    "selection_loss": float(selection_losses[chosen]),
                    "verified_loss": verified_loss,
                    "uniform_verified_loss": uniform_loss,
                    "verified_improvement": uniform_loss - verified_loss,
                    **{f"weight_{SOURCE_NAMES[index]}": float(value) for index, value in enumerate(mixture)},
                }
            )

    diagnostic_df = pd.DataFrame(diagnostics)
    protocol_contrast_rows: list[dict[str, object]] = []
    for metric in [
        "retained_dimension",
        "subspace_overlap",
        "max_principal_angle",
        "top_direction_cosine",
    ]:
        protocol_pivot = diagnostic_df.pivot(index="repeat", columns="protocol", values=metric)
        differences = protocol_pivot["crn"] - protocol_pivot["independent"]
        mean, sd, low, high = t_interval(differences.to_numpy())
        protocol_contrast_rows.append(
            {
                "metric": metric,
                "contrast": "crn_minus_independent",
                "mean_difference": mean,
                "sd_difference": sd,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    protocol_contrast_df = pd.DataFrame(protocol_contrast_rows)
    search_df = pd.DataFrame(search_rows)
    summary_rows: list[dict[str, object]] = []
    for method, subset in search_df.groupby("method"):
        mean, sd, low, high = t_interval(subset.verified_improvement.to_numpy())
        summary_rows.append(
            {
                "method": method,
                "n_search_repeats": len(subset),
                "mean_improvement": mean,
                "sd_improvement": sd,
                "ci95_low": low,
                "ci95_high": high,
                "mean_verified_loss": float(subset.verified_loss.mean()),
                **{
                    f"mean_weight_{name}": float(subset[f"weight_{name}"].mean())
                    for name in SOURCE_NAMES
                },
            }
        )
    search_summary = pd.DataFrame(summary_rows)
    search_summary["method_order"] = search_summary.method.map({method: index for index, method in enumerate(METHODS)})
    search_summary = search_summary.sort_values("method_order").drop(columns="method_order")

    contrast_rows: list[dict[str, object]] = []
    pivot = search_df.pivot(index="repeat", columns="method", values="verified_improvement")
    for other in ["random", "coordinate", "gas_independent"]:
        differences = pivot["gas_crn"] - pivot[other]
        mean, sd, low, high = t_interval(differences.to_numpy())
        contrast_rows.append(
            {
                "contrast": f"gas_crn_minus_{other}",
                "mean_difference": mean,
                "sd_difference": sd,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    contrast_df = pd.DataFrame(contrast_rows)

    source_rows = []
    for index, name in enumerate(SOURCE_NAMES):
        source_rows.append(
            {
                "source": name,
                "n_samples": len(data.x_sources[index]),
                "normalized_label_mean": float(data.y_sources[index].mean()),
                "normalized_label_sd": float(data.y_sources[index].std(ddof=1)),
            }
        )
    pd.DataFrame(source_rows).to_csv(out_dir / "data_source_summary.csv", index=False)
    pd.DataFrame(
        {
            "eigenvalue_index": np.arange(1, len(reference_eigenvalues) + 1),
            "eigenvalue": reference_eigenvalues,
            "energy_fraction": reference_eigenvalues / reference_eigenvalues.sum(),
            "cumulative_energy": np.cumsum(reference_eigenvalues) / reference_eigenvalues.sum(),
        }
    ).to_csv(out_dir / "reference_activity_spectrum.csv", index=False)
    diagnostic_df.to_csv(out_dir / "gas_protocol_diagnostics.csv", index=False)
    protocol_contrast_df.to_csv(out_dir / "gas_protocol_contrasts.csv", index=False)
    search_df.to_csv(out_dir / "mixture_search_by_repeat.csv", index=False)
    search_summary.to_csv(out_dir / "mixture_search_summary.csv", index=False)
    contrast_df.to_csv(out_dir / "mixture_search_contrasts.csv", index=False)
    make_plot(
        reference_eigenvalues,
        diagnostic_df,
        search_summary,
        out_dir / "data_mixture_pilot_summary.png",
    )

    metadata = {
        "n_sources": len(SOURCE_NAMES),
        "source_names": SOURCE_NAMES,
        "mixture_parameterization": "orthonormal centered log-ratio coordinates mapped from [0,1]^(K-1)",
        "training_model": "one-hidden-layer ReLU MLP",
        "training_steps_per_objective_evaluation": training_steps,
        "reference_gas": {"m": reference_m, "q": reference_q, "evaluations": reference_evals},
        "low_budget_gas": {"m": low_m, "q": low_q, "evaluations": low_m * (1 + 2 * low_q * (len(SOURCE_NAMES) - 1))},
        "sensitivity_repeats": sensitivity_repeats,
        "search_repeats": search_repeats,
        "candidate_budget_per_method": candidate_budget,
        "candidate_direction_policy": "test learned basis axes at full and half radius, then fill remaining pairs with random subspace combinations",
        "selection_training_seeds_per_candidate": selection_seeds,
        "verification_training_seeds_per_selected_mixture": verification_seeds,
        "search_radius_in_log_ratio_coordinates": search_radius,
        "statistical_unit_for_search_intervals": "independent candidate-search repeat",
        "runtime_seconds": time.perf_counter() - started,
        "objective_cache_entries": len(objective.cache),
    }
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nProtocol diagnostics:")
    print(diagnostic_df.groupby("protocol").mean(numeric_only=True).to_string())
    print("\nMixture search summary:")
    print(search_summary.to_string(index=False))
    print("\nProtocol contrasts:")
    print(protocol_contrast_df.to_string(index=False))
    print("\nPaired contrasts:")
    print(contrast_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--training-steps", type=int, default=120)
    parser.add_argument("--reference-m", type=int, default=24)
    parser.add_argument("--reference-q", type=int, default=3)
    parser.add_argument("--low-m", type=int, default=8)
    parser.add_argument("--low-q", type=int, default=2)
    parser.add_argument("--sensitivity-repeats", type=int, default=5)
    parser.add_argument("--search-repeats", type=int, default=6)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--selection-seeds", type=int, default=2)
    parser.add_argument("--verification-seeds", type=int, default=12)
    parser.add_argument("--search-radius", type=float, default=0.32)
    args = parser.parse_args()
    run(
        args.out,
        training_steps=args.training_steps,
        reference_m=args.reference_m,
        reference_q=args.reference_q,
        low_m=args.low_m,
        low_q=args.low_q,
        sensitivity_repeats=args.sensitivity_repeats,
        search_repeats=args.search_repeats,
        candidate_budget=args.candidate_budget,
        selection_seeds=args.selection_seeds,
        verification_seeds=args.verification_seeds,
        search_radius=args.search_radius,
    )
