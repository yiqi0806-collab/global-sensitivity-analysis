"""Experiment 8: analytic validation of stochastic GAS estimands.

The experiment separates two scientifically different Global Activity
Subspaces for a stochastic black-box response F(x, xi):

    C_mu   = E[ E(Q | X,V) E(Q | X,V)^T ]
    C_path = E[ Q Q^T ] = C_mu + C_rand.

Four protocols are compared:

1. one stochastic realization per outer anchor with CRN, targeting C_path;
2. K-realization averaging before the outer product, targeting
   C_mu + C_rand / K;
3. a cross-realization U-statistic, targeting C_mu;
4. independent baseline and perturbation states, used as a negative control.

The analytic benchmark is

    F(x, xi) = sum_i alpha_i x_i^2 + tau * xi * b^T x,

with X_i and V_i sampled from Uniform(-1, 1). Its coupled secants and exact
activity matrices are available in closed form, so estimator bias and variance
can be measured without a numerical reference.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "stochastic_estimands"
DEFAULT_SEED = 20261104
DEFAULT_ALPHA = np.array([3.0, 2.0, 1.0, 0.5, 0.25, 0.125])
DEFAULT_B = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]) / math.sqrt(2.0)
NUMERICAL_SEPARATION_FLOOR = 1.0e-10


@dataclass(frozen=True)
class StochasticQuadraticResponse:
    alpha: np.ndarray
    b: np.ndarray
    tau: float

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if alpha.ndim != 1 or b.shape != alpha.shape:
            raise ValueError("alpha and b must be one-dimensional arrays of equal length")
        if not np.isclose(np.linalg.norm(b), 1.0):
            raise ValueError("b must be a unit vector")
        if self.tau < 0:
            raise ValueError("tau must be nonnegative")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "b", b)

    @property
    def dimension(self) -> int:
        return int(self.alpha.size)

    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xi = np.asarray(xi, dtype=float)
        deterministic = np.sum(self.alpha * x * x, axis=-1)
        stochastic_scale = np.einsum("...d,d->...", x, self.b)
        return deterministic + self.tau * stochastic_scale * xi

    def exact_matrices(
        self, min_separation: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return C_mu, C_rand, and C_path for the sampled replacement design.

        When pairs (X_i,V_i) are jointly conditioned on |V_i-X_i| >= h on
        [-1,1]^2, E[(X_i+V_i)^2] = (2-h)^2 / 6.
        """
        if not 0.0 <= min_separation < 2.0:
            raise ValueError("min_separation must lie in [0, 2)")
        second_moment = (2.0 - min_separation) ** 2 / 6.0
        c_mu = second_moment * np.diag(self.alpha**2)
        c_rand = (self.tau**2) * np.outer(self.b, self.b)
        return c_mu, c_rand, c_mu + c_rand

    def exact_k_average_matrix(
        self, k: int, min_separation: float = 0.0
    ) -> np.ndarray:
        if k < 1:
            raise ValueError("k must be positive")
        c_mu, c_rand, _ = self.exact_matrices(min_separation)
        return c_mu + c_rand / k


def seeded_rng(base_seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base_seed, *parts]))


def sample_design(
    n: int,
    d: int,
    rng: np.random.Generator,
    min_separation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample iid coordinate pairs conditioned on a minimum separation.

    Both X and V are redrawn for rejected entries. This produces the joint
    conditional design used by the analytic matrix formula.
    """
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")
    if not 0.0 <= min_separation < 2.0:
        raise ValueError("min_separation must lie in [0, 2)")
    threshold = max(min_separation, NUMERICAL_SEPARATION_FLOOR)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    v = rng.uniform(-1.0, 1.0, size=(n, d))
    invalid = np.abs(v - x) < threshold
    while np.any(invalid):
        count = int(invalid.sum())
        x[invalid] = rng.uniform(-1.0, 1.0, size=count)
        v[invalid] = rng.uniform(-1.0, 1.0, size=count)
        invalid = np.abs(v - x) < threshold
    return x, v


def coupled_secants(
    model: StochasticQuadraticResponse,
    x: np.ndarray,
    v: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """Numerically evaluate CRN secants with shape (outer, K, dimension)."""
    n, d = x.shape
    if v.shape != x.shape or xi.ndim != 2 or xi.shape[0] != n:
        raise ValueError("incompatible x, v, and xi shapes")
    k = xi.shape[1]
    base_inputs = np.repeat(x[:, None, :], k, axis=1)
    baseline = model.evaluate(base_inputs, xi)
    delta = v - x
    q = np.empty((n, k, d), dtype=float)
    for j in range(d):
        perturbed = base_inputs.copy()
        perturbed[:, :, j] = v[:, j, None]
        q[:, :, j] = (
            model.evaluate(perturbed, xi) - baseline
        ) / delta[:, j, None]
    return q


def independent_state_secants(
    model: StochasticQuadraticResponse,
    x: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Use independent random states for every baseline-perturbation pair."""
    n, d = x.shape
    xi_baseline = rng.standard_normal((n, d))
    xi_perturbed = rng.standard_normal((n, d))
    base_inputs = np.repeat(x[:, None, :], d, axis=1)
    perturbed_inputs = base_inputs.copy()
    for j in range(d):
        perturbed_inputs[:, j, j] = v[:, j]
    baseline = model.evaluate(base_inputs, xi_baseline)
    perturbed = model.evaluate(perturbed_inputs, xi_perturbed)
    return (perturbed - baseline) / (v - x)


def outer_mean(q: np.ndarray) -> np.ndarray:
    matrix = np.einsum("nd,ne->de", q, q) / len(q)
    return 0.5 * (matrix + matrix.T)


def k_average_matrix(q: np.ndarray, k: int) -> np.ndarray:
    q_bar = q[:, :k].mean(axis=1)
    return outer_mean(q_bar)


def cross_seed_u_statistic(q: np.ndarray, k: int) -> np.ndarray:
    if k < 2:
        raise ValueError("The cross-seed U-statistic requires k >= 2")
    selected = q[:, :k]
    summed = selected.sum(axis=1)
    outer_sum = np.einsum("nd,ne->nde", summed, summed)
    diagonal_sum = np.einsum("nkd,nke->nde", selected, selected)
    per_anchor = (outer_sum - diagonal_sum) / (k * (k - 1))
    matrix = per_anchor.mean(axis=0)
    return 0.5 * (matrix + matrix.T)


def project_psd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return 0.5 * (projected + projected.T)


def sorted_eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def effective_dimension(matrix: np.ndarray, fraction: float = 0.90) -> int:
    eigenvalues, _ = sorted_eigendecomposition(matrix)
    positive = np.maximum(eigenvalues, 0.0)
    total = float(positive.sum())
    if total <= 0.0:
        return int(len(eigenvalues))
    return int(np.searchsorted(np.cumsum(positive) / total, fraction) + 1)


def relative_norm(error: np.ndarray, target: np.ndarray, ord_value: str) -> float:
    numerator = float(np.linalg.norm(error, ord=ord_value))
    denominator = float(np.linalg.norm(target, ord=ord_value))
    return numerator / max(denominator, np.finfo(float).eps)


def leading_axis_cosine(matrix: np.ndarray, target: np.ndarray) -> float:
    _, vectors = sorted_eigendecomposition(matrix)
    _, target_vectors = sorted_eigendecomposition(target)
    return float(abs(vectors[:, 0] @ target_vectors[:, 0]))


def matrix_metrics(
    estimate: np.ndarray,
    c_mu: np.ndarray,
    c_path: np.ndarray,
    intended: np.ndarray,
) -> dict[str, float]:
    estimate = 0.5 * (estimate + estimate.T)
    eigenvalues, _ = sorted_eigendecomposition(estimate)
    mu_values, _ = sorted_eigendecomposition(c_mu)
    path_values, _ = sorted_eigendecomposition(c_path)
    intended_values, _ = sorted_eigendecomposition(intended)
    projected = project_psd(estimate)

    def eigen_error(target_values: np.ndarray) -> float:
        denominator = max(float(np.linalg.norm(target_values)), np.finfo(float).eps)
        return float(np.linalg.norm(eigenvalues - target_values) / denominator)

    cos_mu = leading_axis_cosine(estimate, c_mu)
    cos_path = leading_axis_cosine(estimate, c_path)
    cos_intended = leading_axis_cosine(estimate, intended)
    return {
        "fro_rel_mu": relative_norm(estimate - c_mu, c_mu, "fro"),
        "fro_rel_path": relative_norm(estimate - c_path, c_path, "fro"),
        "fro_rel_intended": relative_norm(estimate - intended, intended, "fro"),
        "spectral_rel_mu": relative_norm(estimate - c_mu, c_mu, 2),
        "spectral_rel_path": relative_norm(estimate - c_path, c_path, 2),
        "spectral_rel_intended": relative_norm(estimate - intended, intended, 2),
        "eigen_rel_mu": eigen_error(mu_values),
        "eigen_rel_path": eigen_error(path_values),
        "eigen_rel_intended": eigen_error(intended_values),
        "leading_cos_mu": cos_mu,
        "leading_cos_path": cos_path,
        "leading_cos_intended": cos_intended,
        "rank1_projector_error_mu": math.sqrt(max(0.0, 1.0 - cos_mu**2)),
        "rank1_projector_error_path": math.sqrt(max(0.0, 1.0 - cos_path**2)),
        "effective_dimension_90": float(effective_dimension(estimate)),
        "trace": float(np.trace(estimate)),
        "min_eigenvalue": float(eigenvalues.min()),
        "negative_eigenvalue_mass": float(np.maximum(-eigenvalues, 0.0).sum()),
        "psd_projection_fro": float(np.linalg.norm(projected - estimate, ord="fro")),
        "fro_rel_mu_psd": relative_norm(projected - c_mu, c_mu, "fro"),
        "leading_cos_mu_psd": leading_axis_cosine(projected, c_mu),
        "log10_fro_rel_mu": math.log10(
            max(relative_norm(estimate - c_mu, c_mu, "fro"), 1.0e-300)
        ),
    }


GROUP_COLUMNS = [
    "study",
    "protocol",
    "target_label",
    "tau",
    "outer_samples",
    "k",
    "min_separation",
    "logical_evaluations",
]

SUMMARY_METRICS = [
    "fro_rel_mu",
    "fro_rel_path",
    "fro_rel_intended",
    "spectral_rel_intended",
    "eigen_rel_intended",
    "leading_cos_mu",
    "leading_cos_path",
    "leading_cos_intended",
    "rank1_projector_error_mu",
    "rank1_projector_error_path",
    "effective_dimension_90",
    "trace",
    "min_eigenvalue",
    "negative_eigenvalue_mass",
    "psd_projection_fro",
    "fro_rel_mu_psd",
    "leading_cos_mu_psd",
    "log10_fro_rel_mu",
    "theoretical_bias_rel_mu",
]


def add_estimate(
    rows: list[dict[str, object]],
    matrix_groups: dict[tuple[object, ...], dict[str, object]],
    *,
    study: str,
    protocol: str,
    target_label: str,
    repeat: int,
    rng_key: str,
    tau: float,
    outer_samples: int,
    k: int,
    min_separation: float,
    logical_evaluations: int,
    estimate: np.ndarray,
    c_mu: np.ndarray,
    c_rand: np.ndarray,
    c_path: np.ndarray,
    intended: np.ndarray,
) -> None:
    if not np.all(np.isfinite(estimate)):
        raise FloatingPointError(f"non-finite estimate for {study}/{protocol}")
    if not np.allclose(estimate, estimate.T, atol=1.0e-10):
        raise AssertionError(f"nonsymmetric estimate for {study}/{protocol}")
    if protocol in {"one_seed_crn", "k_seed_average"}:
        theoretical_bias = float(
            np.linalg.norm(c_rand / k, ord="fro")
            / max(np.linalg.norm(c_mu, ord="fro"), np.finfo(float).eps)
        )
    elif protocol == "cross_seed_u_stat":
        theoretical_bias = 0.0
    else:
        theoretical_bias = math.nan
    row: dict[str, object] = {
        "study": study,
        "protocol": protocol,
        "target_label": target_label,
        "repeat": repeat,
        "rng_key": rng_key,
        "tau": tau,
        "outer_samples": outer_samples,
        "k": k,
        "min_separation": min_separation,
        "logical_evaluations": logical_evaluations,
        "theoretical_bias_rel_mu": theoretical_bias,
    }
    row.update(matrix_metrics(estimate, c_mu, c_path, intended))
    rows.append(row)

    key = tuple(row[column] for column in GROUP_COLUMNS)
    if key not in matrix_groups:
        matrix_groups[key] = {
            "estimates": [],
            "c_mu": c_mu,
            "c_path": c_path,
            "intended": intended,
        }
    matrix_groups[key]["estimates"].append(estimate)


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, subset in frame.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        row = dict(zip(GROUP_COLUMNS, key))
        row["repeats"] = int(len(subset))
        for metric in SUMMARY_METRICS:
            values = subset[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                for suffix in ["mean", "sd", "median", "q10", "q25", "q75", "q90", "ci95_low", "ci95_high"]:
                    row[f"{metric}_{suffix}"] = math.nan
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_sd": sd,
                    f"{metric}_median": float(np.median(values)),
                    f"{metric}_q10": float(np.quantile(values, 0.10)),
                    f"{metric}_q25": float(np.quantile(values, 0.25)),
                    f"{metric}_q75": float(np.quantile(values, 0.75)),
                    f"{metric}_q90": float(np.quantile(values, 0.90)),
                    f"{metric}_ci95_low": mean - half_width,
                    f"{metric}_ci95_high": mean + half_width,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bias_variance(
    matrix_groups: dict[tuple[object, ...], dict[str, object]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, payload in matrix_groups.items():
        matrices = np.stack(payload["estimates"])
        mean_matrix = matrices.mean(axis=0)
        c_mu = payload["c_mu"]
        c_path = payload["c_path"]
        intended = payload["intended"]
        centered = matrices - mean_matrix
        variance = float(np.mean(np.sum(centered * centered, axis=(1, 2))))

        def decomposition(target: np.ndarray, label: str) -> dict[str, float]:
            scale = max(float(np.linalg.norm(target, ord="fro") ** 2), np.finfo(float).eps)
            bias_sq = float(np.linalg.norm(mean_matrix - target, ord="fro") ** 2)
            mse = float(np.mean(np.sum((matrices - target) ** 2, axis=(1, 2))))
            return {
                f"bias_fro_rel_{label}": math.sqrt(bias_sq / scale),
                f"variance_rel_{label}": variance / scale,
                f"mse_rel_{label}": mse / scale,
                f"decomposition_residual_{label}": mse / scale - (bias_sq + variance) / scale,
            }

        row = dict(zip(GROUP_COLUMNS, key))
        row["repeats"] = int(len(matrices))
        row.update(decomposition(c_mu, "mu"))
        row.update(decomposition(c_path, "path"))
        row.update(decomposition(intended, "intended"))
        row["mean_matrix_leading_cos_mu"] = leading_axis_cosine(mean_matrix, c_mu)
        row["mean_matrix_leading_cos_path"] = leading_axis_cosine(mean_matrix, c_path)
        rows.append(row)
    result = pd.DataFrame(rows)
    for label in ["mu", "path", "intended"]:
        residual = result[f"decomposition_residual_{label}"].abs()
        numerical_scale = np.maximum(result[f"mse_rel_{label}"].abs(), 1.0)
        if float((residual / numerical_scale).max()) > 1.0e-10:
            raise AssertionError("bias-variance decomposition failed")
    return result


def analytic_target_tables(
    amplitudes: list[float],
    separations: list[float],
    k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_rows: list[dict[str, object]] = []
    eigen_rows: list[dict[str, object]] = []
    eigenvector_rows: list[dict[str, object]] = []
    for tau in sorted(set(amplitudes)):
        model = StochasticQuadraticResponse(DEFAULT_ALPHA, DEFAULT_B, tau)
        for separation in sorted(set(separations)):
            c_mu, c_rand, c_path = model.exact_matrices(separation)
            targets: list[tuple[str, int, np.ndarray]] = [
                ("C_mu", 0, c_mu),
                ("C_rand", 0, c_rand),
                ("C_path", 1, c_path),
            ]
            targets.extend(
                ("C_K", k, c_mu + c_rand / k) for k in sorted(set(k_values))
            )
            for label, k, matrix in targets:
                eigenvalues, eigenvectors = sorted_eigendecomposition(matrix)
                for eigen_index in range(model.dimension):
                    eigen_rows.append(
                        {
                            "tau": tau,
                            "min_separation": separation,
                            "target": label,
                            "k": k,
                            "eigen_index": eigen_index + 1,
                            "eigenvalue": float(eigenvalues[eigen_index]),
                        }
                    )
                    for coordinate_index in range(model.dimension):
                        loading = float(eigenvectors[coordinate_index, eigen_index])
                        eigenvector_rows.append(
                            {
                                "tau": tau,
                                "min_separation": separation,
                                "target": label,
                                "k": k,
                                "eigen_index": eigen_index + 1,
                                "coordinate_index": coordinate_index + 1,
                                "loading": loading,
                                "absolute_loading": abs(loading),
                            }
                        )
                for i in range(model.dimension):
                    for j in range(model.dimension):
                        matrix_rows.append(
                            {
                                "tau": tau,
                                "min_separation": separation,
                                "target": label,
                                "k": k,
                                "row": i + 1,
                                "column": j + 1,
                                "value": float(matrix[i, j]),
                            }
                        )
    return (
        pd.DataFrame(matrix_rows),
        pd.DataFrame(eigen_rows),
        pd.DataFrame(eigenvector_rows),
    )


def validate_numeric_secants(
    model: StochasticQuadraticResponse,
    x: np.ndarray,
    v: np.ndarray,
    xi: np.ndarray,
    q: np.ndarray,
) -> float:
    analytic = (
        model.alpha[None, None, :] * (x + v)[:, None, :]
        + model.tau * xi[:, :, None] * model.b[None, None, :]
    )
    max_error = float(np.max(np.abs(q - analytic)))
    if max_error > 5.0e-6:
        raise AssertionError(f"numeric secant validation failed: {max_error:.3e}")
    return max_error


def make_plot(summary: pd.DataFrame, out: Path, main_tau: float) -> None:
    image = Image.new("RGB", (2100, 1320), "white")
    draw = ImageDraw.Draw(image)
    navy = "#17365D"
    blue = "#4F81BD"
    orange = "#F28E2B"
    green = "#59A14F"
    red = "#D62728"
    gray = "#777777"
    colors = {
        "K-average": blue,
        "U-statistic": orange,
        "theoretical 1/K bias": gray,
        "one-seed CRN": green,
        "independent states": red,
    }

    draw.text((70, 35), "Experiment 8: which stochastic GAS matrix is estimated?", fill=navy, font=font(34, True))
    draw.text(
        (72, 86),
        "Analytic stochastic quadratic benchmark; all curves use exact C_mu and C_path targets.",
        fill="#555555",
        font=font(18),
    )

    def panel(
        box: tuple[int, int, int, int],
        title: str,
        x_label: str,
        y_label: str,
        x_values: list[float],
        series: list[tuple[str, list[float]]],
        *,
        y_log: bool = False,
    ) -> None:
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        draw.text((left, top - 55), title, fill=navy, font=font(20, True))
        draw.line((left, top, left, bottom), fill="#333333", width=2)
        draw.line((left, bottom, right, bottom), fill="#333333", width=2)
        finite_y = [value for _, values in series for value in values if np.isfinite(value) and value > 0]
        if not finite_y:
            return
        if y_log:
            y_min = min(finite_y)
            y_max = max(finite_y)
            low = math.floor(math.log10(y_min))
            high = math.ceil(math.log10(y_max))
            if high == low:
                high += 1

            def y_position(value: float) -> int:
                scaled = (math.log10(max(value, 10.0**low)) - low) / (high - low)
                return bottom - int(height * scaled)

            for exponent in range(low, high + 1):
                yy = y_position(10.0**exponent)
                draw.line((left - 6, yy, left, yy), fill="#333333", width=2)
                draw.text((left - 90, yy - 10), f"1e{exponent}", fill="#555555", font=font(13))
        else:
            y_min = 0.0
            y_max = max(finite_y) * 1.08

            def y_position(value: float) -> int:
                return bottom - int(height * (value - y_min) / max(y_max - y_min, 1.0e-12))

            for tick in range(6):
                value = y_min + (y_max - y_min) * tick / 5
                yy = y_position(value)
                draw.line((left - 6, yy, left, yy), fill="#333333", width=2)
                draw.text((left - 75, yy - 10), f"{value:.2f}", fill="#555555", font=font(13))

        x_min, x_max = min(x_values), max(x_values)

        def x_position(value: float) -> int:
            if x_max == x_min:
                return left + width // 2
            return left + int(width * (value - x_min) / (x_max - x_min))

        for value in x_values:
            xx = x_position(value)
            draw.line((xx, bottom, xx, bottom + 6), fill="#333333", width=2)
            label = f"{value:g}"
            draw.text((xx - 14, bottom + 12), label, fill="#555555", font=font(13))
        draw.text((left + width // 2 - 55, bottom + 48), x_label, fill="#555555", font=font(15))
        draw.text((left - 10, top - 25), y_label, fill="#555555", font=font(14))

        for label, values in series:
            points = []
            for x_value, y_value in zip(x_values, values):
                if not np.isfinite(y_value) or y_value <= 0:
                    continue
                xx, yy = x_position(x_value), y_position(y_value)
                points.append((xx, yy))
                draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=colors[label])
            if len(points) >= 2:
                draw.line(points, fill=colors[label], width=4)
        legend_x, legend_y = left + 20, top + 20
        for index, (label, _) in enumerate(series):
            yy = legend_y + 30 * index
            draw.line((legend_x, yy, legend_x + 36, yy), fill=colors[label], width=4)
            draw.text((legend_x + 46, yy - 10), label, fill="#222222", font=font(14))

    k_subset = summary[(summary.study == "k_curve") & (summary.tau == main_tau)]
    k_values = sorted(k_subset.k.unique().tolist())
    k_avg = k_subset[k_subset.protocol == "k_seed_average"].set_index("k")
    u_stat = k_subset[k_subset.protocol == "cross_seed_u_stat"].set_index("k")
    panel(
        (125, 240, 965, 610),
        f"A. Error to C_mu as K increases (tau={main_tau:g})",
        "K stochastic realizations per anchor",
        "relative Frobenius error",
        k_values,
        [
            ("K-average", [float(k_avg.loc[k, "fro_rel_mu_mean"]) for k in k_values]),
            ("U-statistic", [float(u_stat.loc[k, "fro_rel_mu_mean"]) if k in u_stat.index else math.nan for k in k_values]),
            ("theoretical 1/K bias", [float(k_avg.loc[k, "theoretical_bias_rel_mu_mean"]) for k in k_values]),
        ],
        y_log=True,
    )
    panel(
        (1175, 240, 2015, 610),
        "B. Recovery of the mean-response leading axis",
        "K stochastic realizations per anchor",
        "absolute cosine with C_mu axis",
        k_values,
        [
            ("K-average", [float(k_avg.loc[k, "leading_cos_mu_mean"]) for k in k_values]),
            ("U-statistic", [float(u_stat.loc[k, "leading_cos_mu_mean"]) if k in u_stat.index else math.nan for k in k_values]),
        ],
    )

    outer_subset = summary[(summary.study == "outer_curve") & (summary.tau == main_tau)]
    outer_values = sorted(outer_subset.outer_samples.unique().tolist())
    outer_one = outer_subset[outer_subset.protocol == "one_seed_crn"].set_index("outer_samples")
    outer_k = outer_subset[outer_subset.protocol == "k_seed_average"].set_index("outer_samples")
    outer_u = outer_subset[outer_subset.protocol == "cross_seed_u_stat"].set_index("outer_samples")
    panel(
        (125, 810, 965, 1180),
        "C. Convergence to each protocol's intended target",
        "outer anchors N",
        "relative Frobenius error",
        outer_values,
        [
            ("one-seed CRN", [float(outer_one.loc[n, "fro_rel_intended_mean"]) for n in outer_values]),
            ("K-average", [float(outer_k.loc[n, "fro_rel_intended_mean"]) for n in outer_values]),
            ("U-statistic", [float(outer_u.loc[n, "fro_rel_intended_mean"]) for n in outer_values]),
        ],
        y_log=True,
    )

    separation_subset = summary[(summary.study == "separation_curve") & (summary.tau == main_tau)]
    separations = sorted(separation_subset.min_separation.unique().tolist())
    sep_crn = separation_subset[separation_subset.protocol == "one_seed_crn"].set_index("min_separation")
    sep_ind = separation_subset[separation_subset.protocol == "independent_seed"].set_index("min_separation")
    panel(
        (1175, 810, 2015, 1180),
        "D. Independent states amplify small replacement distances",
        "minimum |V_i-X_i|",
        "median error relative to C_mu",
        separations,
        [
            ("one-seed CRN", [float(sep_crn.loc[h, "fro_rel_mu_median"]) for h in separations]),
            ("independent states", [float(sep_ind.loc[h, "fro_rel_mu_median"]) for h in separations]),
        ],
        y_log=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def run(
    out_dir: Path,
    *,
    repeats: int = 200,
    k_outer_samples: int = 256,
    k_values: list[int] | None = None,
    outer_sizes: list[int] | None = None,
    amplitudes: list[float] | None = None,
    main_amplitude: float = 3.0,
    main_k: int = 8,
    separations: list[float] | None = None,
    fixed_budget_secants: int = 4096,
    seed: int = DEFAULT_SEED,
) -> None:
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32]
    if outer_sizes is None:
        outer_sizes = [64, 128, 256, 512, 1024]
    if amplitudes is None:
        amplitudes = [0.0, 1.0, 2.5, 3.0]
    if separations is None:
        separations = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    if any(k < 1 for k in k_values) or 1 not in k_values:
        raise ValueError("k_values must be positive and include 1")
    if main_k < 2:
        raise ValueError("main_k must be at least 2")
    if main_k not in k_values:
        k_values = sorted(set([*k_values, main_k]))
    if main_amplitude not in amplitudes:
        amplitudes = sorted(set([*amplitudes, main_amplitude]))
    if any(size <= 0 for size in outer_sizes):
        raise ValueError("outer sample sizes must be positive")
    if any(fixed_budget_secants % k for k in k_values):
        raise ValueError("fixed_budget_secants must be divisible by every k value")

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    matrix_groups: dict[tuple[object, ...], dict[str, object]] = {}
    max_k = max(k_values)
    max_outer = max(outer_sizes)
    d = len(DEFAULT_ALPHA)
    max_numeric_secant_error = 0.0

    # Study 1: fixed outer sample size, increasing K and stochastic amplitude.
    for repeat in range(repeats):
        for amplitude_index, tau in enumerate(amplitudes):
            rng_key = f"{seed}:1:{repeat}:{amplitude_index}"
            rng = seeded_rng(seed, 1, repeat, amplitude_index)
            model = StochasticQuadraticResponse(DEFAULT_ALPHA, DEFAULT_B, tau)
            x, v = sample_design(k_outer_samples, d, rng, 0.0)
            xi = rng.standard_normal((k_outer_samples, max_k))
            q = coupled_secants(model, x, v, xi)
            if repeat == 0:
                max_numeric_secant_error = max(
                    max_numeric_secant_error,
                    validate_numeric_secants(model, x, v, xi, q),
                )
            c_mu, c_rand, c_path = model.exact_matrices(0.0)
            one_seed = outer_mean(q[:, 0, :])
            add_estimate(
                rows,
                matrix_groups,
                study="k_curve",
                protocol="one_seed_crn",
                target_label="C_path",
                repeat=repeat,
                rng_key=rng_key,
                tau=tau,
                outer_samples=k_outer_samples,
                k=1,
                min_separation=0.0,
                logical_evaluations=k_outer_samples * (d + 1),
                estimate=one_seed,
                c_mu=c_mu,
                c_rand=c_rand,
                c_path=c_path,
                intended=c_path,
            )
            for k in k_values:
                averaged = k_average_matrix(q, k)
                if k == 1 and not np.allclose(averaged, one_seed, atol=1.0e-12):
                    raise AssertionError("K=1 average does not equal one-seed CRN")
                c_k = c_mu + c_rand / k
                add_estimate(
                    rows,
                    matrix_groups,
                    study="k_curve",
                    protocol="k_seed_average",
                    target_label="C_mu_plus_C_rand_over_K",
                    repeat=repeat,
                    rng_key=rng_key,
                    tau=tau,
                    outer_samples=k_outer_samples,
                    k=k,
                    min_separation=0.0,
                    logical_evaluations=k_outer_samples * k * (d + 1),
                    estimate=averaged,
                    c_mu=c_mu,
                    c_rand=c_rand,
                    c_path=c_path,
                    intended=c_k,
                )
                if k >= 2:
                    u_stat = cross_seed_u_statistic(q, k)
                    add_estimate(
                        rows,
                        matrix_groups,
                        study="k_curve",
                        protocol="cross_seed_u_stat",
                        target_label="C_mu",
                        repeat=repeat,
                        rng_key=rng_key,
                        tau=tau,
                        outer_samples=k_outer_samples,
                        k=k,
                        min_separation=0.0,
                        logical_evaluations=k_outer_samples * k * (d + 1),
                        estimate=u_stat,
                        c_mu=c_mu,
                        c_rand=c_rand,
                        c_path=c_path,
                        intended=c_mu,
                    )

    # Study 2: outer-anchor convergence at a fixed K and amplitude.
    for repeat in range(repeats):
        rng_key = f"{seed}:2:{repeat}"
        rng = seeded_rng(seed, 2, repeat)
        model = StochasticQuadraticResponse(DEFAULT_ALPHA, DEFAULT_B, main_amplitude)
        x, v = sample_design(max_outer, d, rng, 0.0)
        xi = rng.standard_normal((max_outer, main_k))
        q = coupled_secants(model, x, v, xi)
        c_mu, c_rand, c_path = model.exact_matrices(0.0)
        c_k = c_mu + c_rand / main_k
        for n in outer_sizes:
            q_n = q[:n]
            for protocol, estimate, intended, target_label, k in [
                ("one_seed_crn", outer_mean(q_n[:, 0, :]), c_path, "C_path", 1),
                ("k_seed_average", k_average_matrix(q_n, main_k), c_k, "C_mu_plus_C_rand_over_K", main_k),
                ("cross_seed_u_stat", cross_seed_u_statistic(q_n, main_k), c_mu, "C_mu", main_k),
            ]:
                add_estimate(
                    rows,
                    matrix_groups,
                    study="outer_curve",
                    protocol=protocol,
                    target_label=target_label,
                    repeat=repeat,
                    rng_key=rng_key,
                    tau=main_amplitude,
                    outer_samples=n,
                    k=k,
                    min_separation=0.0,
                    logical_evaluations=n * k * (d + 1),
                    estimate=estimate,
                    c_mu=c_mu,
                    c_rand=c_rand,
                    c_path=c_path,
                    intended=intended,
                )

    # Study 3: fixed total number of coupled secant vectors N*K.
    for repeat in range(repeats):
        for k in k_values:
            n = fixed_budget_secants // k
            rng_key = f"{seed}:3:{repeat}:{k}"
            rng = seeded_rng(seed, 3, repeat, k)
            model = StochasticQuadraticResponse(DEFAULT_ALPHA, DEFAULT_B, main_amplitude)
            x, v = sample_design(n, d, rng, 0.0)
            xi = rng.standard_normal((n, k))
            q = coupled_secants(model, x, v, xi)
            c_mu, c_rand, c_path = model.exact_matrices(0.0)
            c_k = c_mu + c_rand / k
            averaged = k_average_matrix(q, k)
            add_estimate(
                rows,
                matrix_groups,
                study="fixed_budget",
                protocol="k_seed_average",
                target_label="C_mu_plus_C_rand_over_K",
                repeat=repeat,
                rng_key=rng_key,
                tau=main_amplitude,
                outer_samples=n,
                k=k,
                min_separation=0.0,
                logical_evaluations=fixed_budget_secants * (d + 1),
                estimate=averaged,
                c_mu=c_mu,
                c_rand=c_rand,
                c_path=c_path,
                intended=c_k,
            )
            if k >= 2:
                u_stat = cross_seed_u_statistic(q, k)
                add_estimate(
                    rows,
                    matrix_groups,
                    study="fixed_budget",
                    protocol="cross_seed_u_stat",
                    target_label="C_mu",
                    repeat=repeat,
                    rng_key=rng_key,
                    tau=main_amplitude,
                    outer_samples=n,
                    k=k,
                    min_separation=0.0,
                    logical_evaluations=fixed_budget_secants * (d + 1),
                    estimate=u_stat,
                    c_mu=c_mu,
                    c_rand=c_rand,
                    c_path=c_path,
                    intended=c_mu,
                )

    # Study 4: independent-state negative control versus replacement separation.
    for repeat in range(repeats):
        for separation_index, separation in enumerate(separations):
            rng_key = f"{seed}:4:{repeat}:{separation_index}"
            rng = seeded_rng(seed, 4, repeat, separation_index)
            model = StochasticQuadraticResponse(DEFAULT_ALPHA, DEFAULT_B, main_amplitude)
            x, v = sample_design(k_outer_samples, d, rng, separation)
            xi = rng.standard_normal((k_outer_samples, 1))
            q_crn = coupled_secants(model, x, v, xi)[:, 0, :]
            q_independent = independent_state_secants(model, x, v, rng)
            c_mu, c_rand, c_path = model.exact_matrices(separation)
            add_estimate(
                rows,
                matrix_groups,
                study="separation_curve",
                protocol="one_seed_crn",
                target_label="C_path",
                repeat=repeat,
                rng_key=rng_key,
                tau=main_amplitude,
                outer_samples=k_outer_samples,
                k=1,
                min_separation=separation,
                logical_evaluations=k_outer_samples * (d + 1),
                estimate=outer_mean(q_crn),
                c_mu=c_mu,
                c_rand=c_rand,
                c_path=c_path,
                intended=c_path,
            )
            add_estimate(
                rows,
                matrix_groups,
                study="separation_curve",
                protocol="independent_seed",
                target_label="negative_control_diagnostic_against_C_mu",
                repeat=repeat,
                rng_key=rng_key,
                tau=main_amplitude,
                outer_samples=k_outer_samples,
                k=1,
                min_separation=separation,
                logical_evaluations=2 * k_outer_samples * d,
                estimate=outer_mean(q_independent),
                c_mu=c_mu,
                c_rand=c_rand,
                c_path=c_path,
                intended=c_mu,
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty or metrics[SUMMARY_METRICS].isna().all(axis=1).any():
        raise AssertionError("missing experiment metrics")
    summary = summarize_metrics(metrics)
    bias_variance = summarize_bias_variance(matrix_groups)
    target_matrices, target_eigenvalues, target_eigenvectors = analytic_target_tables(
        sorted(set([*amplitudes, main_amplitude])),
        sorted(set([0.0, *separations])),
        k_values,
    )

    metrics.to_csv(out_dir / "estimator_metrics_by_repeat.csv", index=False)
    summary.to_csv(out_dir / "estimator_metrics_summary.csv", index=False)
    bias_variance.to_csv(out_dir / "bias_variance_summary.csv", index=False)
    target_matrices.to_csv(out_dir / "analytic_target_matrices.csv", index=False)
    target_eigenvalues.to_csv(out_dir / "analytic_target_eigenvalues.csv", index=False)
    target_eigenvectors.to_csv(out_dir / "analytic_target_eigenvectors.csv", index=False)
    make_plot(summary, out_dir / "stochastic_estimands_summary.png", main_amplitude)
    make_plot(summary, ROOT / "figures" / "stochastic_estimands_summary.png", main_amplitude)

    elapsed = time.perf_counter() - started
    metadata = {
        "experiment": "Experiment 8 - stochastic GAS estimand validation",
        "benchmark": "F(x,xi)=sum_i alpha_i*x_i^2 + tau*xi*b^T*x",
        "alpha": DEFAULT_ALPHA.tolist(),
        "b": DEFAULT_B.tolist(),
        "input_and_replacement_domain": "[-1,1]^d",
        "dimension": d,
        "random_state_distribution": "standard normal",
        "repeats": repeats,
        "k_outer_samples": k_outer_samples,
        "k_values": k_values,
        "outer_sizes": outer_sizes,
        "amplitudes": amplitudes,
        "main_amplitude": main_amplitude,
        "main_k": main_k,
        "min_separations": separations,
        "fixed_budget_secant_vectors": fixed_budget_secants,
        "base_seed": seed,
        "rng_key_format": "base_seed:study_id:repeat[:parameter_index]",
        "one_seed_interpretation": "one independent stochastic realization per outer anchor",
        "coupling": "baseline and all coordinate replacements at an anchor share xi",
        "independent_control": "each coordinate uses independent baseline and perturbed states",
        "replacement_sampling_design": "X and V are jointly redrawn until each coordinate pair satisfies the stated minimum separation",
        "logical_evaluation_cost": "per-estimator black-box call count; nested K and N reuse makes total script runtime smaller than the sum of these costs",
        "u_statistic_psd_projection": "per-repeat error diagnostic only; the raw matrix is the unbiased estimator",
        "max_numeric_secant_validation_error": max_numeric_secant_error,
        "runtime_seconds": elapsed,
        "result_rows": int(len(metrics)),
    }
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    key_results = summary[
        (summary.study == "k_curve")
        & (summary.tau == main_amplitude)
        & (summary.k.isin([1, 2, 4, 8, 16, 32]))
        & (summary.protocol.isin(["k_seed_average", "cross_seed_u_stat"]))
    ][
        [
            "protocol",
            "k",
            "fro_rel_mu_mean",
            "leading_cos_mu_mean",
            "theoretical_bias_rel_mu_mean",
            "negative_eigenvalue_mass_mean",
        ]
    ]
    print(json.dumps(metadata, indent=2))
    print("\nMain K-curve results:")
    print(key_results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--k-outer-samples", type=int, default=256)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--outer-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.0, 1.0, 2.5, 3.0])
    parser.add_argument("--main-amplitude", type=float, default=3.0)
    parser.add_argument("--main-k", type=int, default=8)
    parser.add_argument("--separations", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--fixed-budget-secants", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run(
        args.out,
        repeats=args.repeats,
        k_outer_samples=args.k_outer_samples,
        k_values=args.k_values,
        outer_sizes=args.outer_sizes,
        amplitudes=args.amplitudes,
        main_amplitude=args.main_amplitude,
        main_k=args.main_k,
        separations=args.separations,
        fixed_budget_secants=args.fixed_budget_secants,
        seed=args.seed,
    )
