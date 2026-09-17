"""Reproducible GAS and generalized Sobol study for Sergei's four-input g-function.

The supplied manual specifies a=[0,1,99,99] and a Gaussian correlation template.
This is an independent Python implementation, not an execution of his MATLAB code.
GAS uses the finite divided-difference matrix in gas_with_copula.ipynb.
GAS labels refer to independent drivers; Sobol labels refer to the target physical inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from numpy.polynomial.legendre import leggauss
from gas_energy import GAS_score_95

A = np.array([0., 1., 99., 99.])
ORDERS = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]


def correlation(rho):
    return np.array([[1., rho, 0., rho], [rho, 1., 0., 0.],
                     [0., 0., 1., 0.], [rho, 0., 0., 1.]])


def transform(z, rho, order):
    L = np.linalg.cholesky(correlation(rho)[np.ix_(order, order)])
    x = np.empty_like(z)
    x[:, order] = ndtr(z @ L.T)
    return x


def model(x):
    return np.prod((np.abs(4*x-2)+A)/(1+A), axis=-1)


def gas_matrix(power, seed, rho, order, coord="uniform", quadrature=16):
    """Split integration at base point and EVERY absolute-value kink.

    E[D_i^2] fills the diagonal; E[E[D_i|z] E[D_j|z]] fills off-diagonals.
    The latter relies on independent replacement coordinates. The independent
    Sobol points cover the entire cube (no 0.01/0.99 truncation).
    """
    base = qmc.Sobol(4, scramble=True, seed=seed).random_base2(power)
    z = ndtri(base)
    L = np.linalg.cholesky(correlation(rho)[np.ix_(order, order)])
    w = z @ L.T
    ap = A[list(order)]
    y = np.prod((np.abs(4*ndtr(w)-2)+ap)/(1+ap), axis=1)
    nodes, weights = leggauss(quadrature)
    nodes, weights = (nodes+1)/2, weights/2
    avg = np.empty_like(base)
    sq = np.empty_like(base)
    for j in range(4):
        knots = [np.zeros(len(z)), np.ones(len(z)), base[:, j]]
        # w_k(v)=w_k + L_kj (Phi^-1(v)-z_j), kink at w_k(v)=0.
        for k in range(4):
            if abs(L[k, j]) > 1e-14:
                knots.append(ndtr(z[:, j]-w[:, k]/L[k, j]))
        knots = np.sort(np.stack(knots, axis=1), axis=1)
        aj, sj = np.zeros(len(z)), np.zeros(len(z))
        for b in range(knots.shape[1]-1):
            lo, hi = knots[:, b], knots[:, b+1]
            width = hi-lo
            v = lo[:, None]+width[:, None]*nodes
            zv = ndtri(np.clip(v, 1e-15, 1-1e-15))
            shifted = w[:, None, :]+(zv-z[:, j, None])[:, :, None]*L[:, j]
            fy = np.prod((np.abs(4*ndtr(shifted)-2)+ap)/(1+ap), axis=2)
            den = (v-base[:, j, None]) if coord == "uniform" else (zv-z[:, j, None])
            diff = np.divide(fy-y[:, None], den, out=np.zeros_like(fy), where=np.abs(den)>1e-14)
            aj += width*(diff @ weights)
            sj += width*((diff*diff) @ weights)
        avg[:, j], sq[:, j] = aj, sj
    C = avg.T @ avg / len(z)
    np.fill_diagonal(C, sq.mean(axis=0))
    eigen, vectors = np.linalg.eigh(C)
    idx = np.argsort(eigen)[::-1]
    eigen, vectors = eigen[idx], vectors[:, idx]
    if eigen.min() < -1e-9:
        raise AssertionError("GAS matrix is not positive semidefinite")
    scores = {}
    for rank in (1, 2, 4):
        raw = (vectors[:, :rank]**2) @ eigen[:rank]
        physical_labels = np.empty(4)
        physical_labels[list(order)] = raw / raw.sum()
        scores[rank] = physical_labels
    mapped_C = np.empty((4, 4))
    mapped_C[np.ix_(order, order)] = C
    return mapped_C, eigen, scores


def sobol_four(power, seed, rho):
    """Standard Sobol estimators of each cyclic inverse Rosenblatt map.

    First driver gives full S and full ST for the first physical variable;
    last driver gives independent S and independent ST for the last variable.
    ST_ind=E Var(Y|X_-i)/Var(Y), unlike the manual's duplicated Eq. (2)/(4).
    First-order covariance estimator is centered to reduce sampling error.
    """
    pair = qmc.Sobol(8, scramble=True, seed=seed).random_base2(power)
    za, zb = ndtri(pair[:, :4]), ndtri(pair[:, 4:])
    rows = []
    for order in ORDERS:
        ya, yb = model(transform(za, rho, order)), model(transform(zb, rho, order))
        mu = (ya.mean()+yb.mean())/2
        var = (np.mean((ya-mu)**2)+np.mean((yb-mu)**2))/2
        for j, family in [(0, "full"), (3, "ind")]:
            zc = za.copy()
            zc[:, j] = zb[:, j]
            yc = model(transform(zc, rho, order))
            first = np.mean((yb-mu)*(yc-ya))/var
            total = np.mean((ya-yc)**2)/(2*var)
            rows.append(dict(rho=rho, variable=order[j]+1, family=family,
                             S=first, ST=total, variance=var, mean=mu))
    return rows


def fixing(power, seed, rho):
    x = transform(ndtri(qmc.Sobol(4, scramble=True, seed=seed).random_base2(power)), rho, ORDERS[0])
    y = model(x)
    var, sd = y.var(), y.std()
    rows = []
    for variables in [(0,), (1,), (2,), (3,), (2, 3)]:
        for value in (.5, .25):
            reduced = x.copy()
            reduced[:, variables] = value
            yr = model(reduced)
            rows.append(dict(rho=rho, fixed="+".join(f"x{i+1}" for i in variables), value=value,
                             mse_over_variance=np.mean((yr-y)**2)/var,
                             rmse_over_sd=np.sqrt(np.mean((yr-y)**2))/sd,
                             mean_bias_over_sd=np.mean(yr-y)/sd,
                             relative_variance_change=yr.var()/var-1))
    return rows


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average(rows, keys, metrics):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, values in grouped.items():
        row = dict(zip(keys, key))
        for metric in metrics:
            a = np.array([v[metric] for v in values])
            row[metric] = a.mean()
            row[metric+"_se"] = a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0.
        out.append(row)
    return out


def validate(out):
    q = 1/(3*(1+A)**2)
    variance = np.prod(1+q)-1
    exact_first = q/variance
    exact_total = q*np.prod(1+q)/(1+q)/variance
    # Same one-dimensional shape yields diagonal GAS proportional to exact ST at rho=0.
    exact_gas = exact_total/exact_total.sum()
    numerical = sobol_four(18, 94811, 0.)
    err = max(max(abs(row["S"]-exact_first[row["variable"]-1]),
                  abs(row["ST"]-exact_total[row["variable"]-1])) for row in numerical)
    C, eigen, scores = gas_matrix(15, 94811, 0., ORDERS[0], quadrature=32)
    gas_err = np.max(np.abs(scores[4]-exact_gas))
    assert err < .003, err
    assert gas_err < .002, gas_err
    # Verify the divided-difference integrator on a linear model by direct
    # independent calculation of its exact constant divided-difference matrix.
    coef = np.array([1., 2., 3., 4.])
    rng = np.random.default_rng(501)
    zz, vv = rng.random((2, 4096, 4))
    dd = np.empty_like(zz)
    for j in range(4):
        ww = zz.copy(); ww[:, j] = vv[:, j]
        dd[:, j] = (ww@coef-zz@coef)/(vv[:, j]-zz[:, j])
    linear_error = np.max(np.abs(dd.T@dd/len(dd)-np.outer(coef,coef)))
    assert linear_error < 1e-9
    result = dict(exact_variance=float(variance), exact_S=exact_first.tolist(),
                  exact_ST=exact_total.tolist(), exact_GAS_rank4=exact_gas.tolist(),
                  sobol_max_abs_error=err, gas_max_abs_error=float(gas_err),
                  numerical_GAS_rank1=scores[1].tolist(), eigenvalues=eigen.tolist(),
                  linear_divided_difference_max_abs_error=float(linear_error))
    (out/"validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Validation:", result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--output", default="source/results/sergei_g_function")
    args = parser.parse_args()
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    gasrows, eigenrows, sobolrows, fixrows = [], [], [], []
    adaptive_rows, selection_rows, cached_matrices = [], [], []
    rhos = [0., .4, .7] if args.pilot or args.convergence else np.round(np.arange(-.7, .701, .1), 2).tolist()
    repeats = 1 if args.pilot else (4 if args.convergence else 6)
    power = 11 if args.pilot else (15 if args.convergence else 13)
    quad = 12 if args.pilot else (40 if args.convergence else 20)
    configs = [(order, "uniform") for order in ORDERS]+[(ORDERS[0], "normal")]
    if args.convergence:
        configs = [(ORDERS[0], "uniform"), (ORDERS[3], "uniform"), (ORDERS[0], "normal")]
    if args.normal_only:
        configs = [(order, "normal") for order in (ORDERS if not args.convergence else [ORDERS[0], ORDERS[3]])]
    for rep in range(repeats):
        for rho in rhos:
            for order, coord in configs:
                C, eigen, scores = gas_matrix(power, 20260916+rep*1009, rho, order, coord, quad)
                ordername = "".join(str(i+1) for i in order)
                # C has already been mapped back to physical-variable labels.
                adaptive_eigen, adaptive_vectors = np.linalg.eigh(C)
                adaptive_scores, m, retained = GAS_score_95(adaptive_vectors, adaptive_eigen)
                previous = float(np.maximum(adaptive_eigen, 0)[::-1][:m-1].sum()/np.maximum(adaptive_eigen,0).sum())
                assert retained >= .95 and previous < .95
                selection_rows.append(dict(rep=rep, rho=rho, order=ordername, coordinate=coord,
                                           selected_rank=m, retained_fraction=retained,
                                           previous_fraction=previous))
                cached_matrices.append(C)
                for i, value in enumerate(adaptive_scores):
                    adaptive_rows.append(dict(rep=rep, rho=rho, order=ordername, coordinate=coord,
                                              variable=i+1, GAS=float(value), selected_rank=m,
                                              retained_fraction=retained))
                for rank, values in scores.items():
                    for i, value in enumerate(values):
                        gasrows.append(dict(rep=rep, rho=rho, order=ordername, coordinate=coord,
                                            rank=rank, variable=i+1, GAS=float(value), Cii=float(C[i,i]),
                                            trace=float(np.trace(C))))
                eigenrows.append(dict(rep=rep, rho=rho, order=ordername, coordinate=coord,
                                      **{f"lambda{k+1}":float(v) for k,v in enumerate(eigen)},
                                      rank1_fraction=float(eigen[0]/eigen.sum()),
                                      rank2_fraction=float(eigen[:2].sum()/eigen.sum())))
            if not args.convergence and not args.normal_only:
                for row in sobol_four(14 if args.pilot else 17, 20310916+rep*1009, rho):
                    sobolrows.append(dict(rep=rep, **row))
                for row in fixing(14 if args.pilot else 18, 20410916+rep*1009, rho):
                    fixrows.append(dict(rep=rep, **row))
            print(f"rep={rep+1}/{repeats} rho={rho:+.1f} elapsed={time.time()-start:.1f}s", flush=True)
        # Preserve completed repetitions in case a long run is interrupted.
        save_csv(out/"gas_replicates.csv", gasrows)
        save_csv(out/"gas_eigenvalues.csv", eigenrows)
        save_csv(out/"gas_summary.csv", average(gasrows, ["rho","order","coordinate","rank","variable"], ["GAS","Cii","trace"]))
        save_csv(out/"gas95_replicates.csv", adaptive_rows)
        save_csv(out/"gas95_selection.csv", selection_rows)
        adaptive_summary = average(adaptive_rows, ["rho","order","coordinate","variable"], ["GAS"])
        for row in adaptive_summary:
            choices = [r for r in selection_rows if all(r[k] == row[k] for k in ["rho","order","coordinate"])]
            row.update(selected_rank_min=min(r["selected_rank"] for r in choices),
                       selected_rank_max=max(r["selected_rank"] for r in choices),
                       retained_fraction_min=min(r["retained_fraction"] for r in choices),
                       retained_fraction_max=max(r["retained_fraction"] for r in choices))
        save_csv(out/"gas95_summary.csv", adaptive_summary)
        np.savez_compressed(out/"gas_matrices.npz", C=np.asarray(cached_matrices),
                            rho=np.array([r["rho"] for r in selection_rows]),
                            rep=np.array([r["rep"] for r in selection_rows]),
                            order=np.array([r["order"] for r in selection_rows]),
                            coordinate=np.array([r["coordinate"] for r in selection_rows]))
        if sobolrows:
            save_csv(out/"sobol_replicates.csv", sobolrows)
            save_csv(out/"sobol_summary.csv", average(sobolrows, ["rho","variable","family"], ["S","ST","variance","mean"]))
            save_csv(out/"fixing_replicates.csv", fixrows)
            save_csv(out/"fixing_summary.csv", average(fixrows, ["rho","fixed","value"], ["mse_over_variance","rmse_over_sd","mean_bias_over_sd","relative_variance_change"]))
    if not args.convergence:
        validate(out)
    metadata = dict(a=A.tolist(), latent_correlation_template=correlation(.5).tolist(),
                    note="Off-diagonal 0.5 entries are replaced by each rho; rho is Gaussian copula correlation.",
                    rhos=rhos, repeats=repeats, gas_outer_N=2**power, quadrature_nodes_per_segment=quad,
                    primary_gas_selection="minimum rank retaining at least 0.95 of estimated trace",
                    sobol_N=2**(14 if args.pilot else 17), fixing_N=2**(14 if args.pilot else 18),
                    gas_seeds=[20260916+r*1009 for r in range(repeats)],
                    elapsed_seconds=time.time()-start, numpy_version=np.__version__)
    (out/"run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
