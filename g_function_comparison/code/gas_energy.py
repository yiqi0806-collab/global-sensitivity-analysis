"""Select the smallest GAS eigenspace retaining a requested trace fraction."""
import numpy as np


def GAS_score_95(u, s, threshold=0.95):
    """Return (normalized scores, selected rank, retained trace fraction).

    Eigenvectors are columns paired with s; input ordering may be arbitrary.
    The fraction concerns the estimated GAS matrix, not output variance or
    accuracy of each individual variable's score. A zero matrix returns zero
    scores, rank 0 and NaN coverage because no importance exists to normalize.
    """
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    s = np.asarray(s, dtype=float)
    u = np.asarray(u, dtype=float)
    if s.ndim != 1 or s.size == 0 or u.shape != (s.size, s.size):
        raise ValueError("Supply all eigenvalues and their matching eigenvectors")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(u)):
        raise ValueError("Eigenpairs must be finite")
    if not np.allclose(u.T @ u, np.eye(s.size), atol=1e-9, rtol=1e-9):
        raise ValueError("Eigenvectors must be orthonormal columns")
    scale = np.max(np.abs(s))
    if np.min(s) < -1e-12 * scale:
        raise ValueError("GAS matrix has a materially negative eigenvalue")
    s = np.maximum(s, 0.)  # Only roundoff-sized negative eigenvalues reach here.
    idx = np.argsort(s)[::-1]
    s, u = s[idx], u[:, idx]
    if s[0] == 0:
        return np.zeros(s.size), 0, float("nan")
    scaled = s / s[0]  # Avoid underflow/overflow in the trace.
    cumulative = np.cumsum(scaled)
    cumulative /= cumulative[-1]
    m = int(np.searchsorted(cumulative, threshold, side="left")) + 1
    raw = (u[:, :m] ** 2) @ scaled[:m]
    return raw / raw.sum(), m, float(cumulative[m-1])
