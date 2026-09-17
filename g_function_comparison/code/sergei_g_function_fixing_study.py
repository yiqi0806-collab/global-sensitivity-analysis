"""Fresh GAS screening and physical-variable fixing runs for the g-function."""
from pathlib import Path
import argparse
import json
import time

from sergei_g_function_study import (
    A, ROOT, ORDERS, np, qmc, ndtri, ndtr, correlation, transform, model,
    gas_matrix, GAS_score_95, sobol_four, fixing, save_csv, average,
)

OUT = ROOT / 'source/results/sergei_g_function_fixing_rerun'
RHOS = np.round(np.arange(-.7, .701, .1), 2).tolist()
SEEDS = {'gas': 7310917, 'sobol': 8310923, 'fixing': 9310931}


def conditional_total(power, seed, rho):
    u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(power)
    w = ndtri(u[:, :4]) @ np.linalg.cholesky(correlation(rho)).T
    extra = ndtri(u[:, 4])
    y = model(ndtr(w))
    precision = np.linalg.inv(correlation(rho))
    result = []
    for i in range(4):
        coefficients = -precision[i] / precision[i, i]
        coefficients[i] = 0
        replaced = w.copy()
        replaced[:, i] = w @ coefficients + extra / np.sqrt(precision[i, i])
        yp = model(ndtr(replaced))
        result.append(np.mean((y-yp)**2)/(2*y.var()))
    return np.asarray(result)


def write_summaries(gas_rows, sobol_rows, fix_rows, matrices):
    save_csv(OUT/'gas_replicates.csv', gas_rows)
    save_csv(OUT/'gas_summary.csv', average(gas_rows, ['rho', 'variable'], ['GAS']))
    save_csv(OUT/'sobol_replicates.csv', sobol_rows)
    save_csv(OUT/'sobol_summary.csv', average(sobol_rows, ['rho','variable','family'], ['S','ST','variance','mean']))
    save_csv(OUT/'fixing_replicates.csv', fix_rows)
    save_csv(OUT/'fixing_summary.csv', average(fix_rows, ['rho','fixed','value'],
             ['mse_over_variance','rmse_over_sd','mean_bias_over_sd','relative_variance_change']))
    np.savez_compressed(OUT/'gas_matrices.npz', C=np.asarray(matrices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repeats = 1 if args.pilot else 6
    rhos = [0., .7] if args.pilot else RHOS
    gas_power, quad = (11, 12) if args.pilot else (14, 30)
    sobol_power, fix_power = (14, 14) if args.pilot else (18, 19)
    grows, srows, frows, matrices = [], [], [], []
    for rep in range(repeats):
        for rho in rhos:
            C, eigen, full_scores = gas_matrix(gas_power, SEEDS['gas']+1009*rep,
                                              rho, ORDERS[0], 'normal', quad)
            eigen, vectors = np.linalg.eigh(C)
            scores, rank, fraction = GAS_score_95(vectors, eigen)
            assert fraction >= .95 and np.all(scores >= 0)
            assert np.isclose(scores.sum(), 1.)
            matrices.append(C)
            for i, score in enumerate(scores):
                grows.append(dict(rep=rep, rho=rho, variable=i+1, GAS=float(score),
                                  selected_rank=rank, retained_fraction=fraction))
            for row in sobol_four(sobol_power, SEEDS['sobol']+1009*rep, rho):
                srows.append(dict(rep=rep, **row))
            for row in fixing(fix_power, SEEDS['fixing']+1009*rep, rho):
                frows.append(dict(rep=rep, **row))
            print(f'Completed repeat {rep+1}/{repeats}, rho={rho:+.1f}, elapsed={time.time()-started:.1f}s', flush=True)
        write_summaries(grows, srows, frows, matrices)

    # Independent analytical checks use the newly generated rho=0 data.
    q = 1/(3*(1+A)**2)
    variance = np.prod(1+q)-1
    exact_S = q/variance
    exact_ST = q*np.prod(1+q)/(1+q)/variance
    sobol_summary = average(srows, ['rho','variable','family'], ['S','ST'])
    sobol_error = max(abs(row[key]-exact[row['variable']-1])
        for row in sobol_summary if row['rho']==0
        for key,exact in [('S',exact_S),('ST',exact_ST)])
    full_C = np.mean([matrices[k] for k in range(len(matrices)) if grows[4*k]['rho']==0], axis=0)
    exact_full_GAS = exact_ST/exact_ST.sum()
    gas_full_error = float(np.max(np.abs(np.diag(full_C)/np.trace(full_C)-exact_full_GAS)))
    fixing_summary = average(frows, ['rho','fixed','value'], ['rmse_over_sd','mse_over_variance'])
    fixing_error = 0.
    for row in fixing_summary:
        if row['rho'] != 0: continue
        indices = [int(name[1:])-1 for name in row['fixed'].split('+')]
        remaining = [i for i in range(4) if i not in indices]
        k = np.prod((np.abs(4*row['value']-2)+A[indices])/(1+A[indices]))
        exact_mse = np.prod(1+q[remaining])*(np.prod(1+q[indices])-2*k+k*k)
        exact_error = np.sqrt(max(0.,exact_mse)/variance)
        fixing_error = max(fixing_error, abs(row['rmse_over_sd']-exact_error))
    assert sobol_error < .001
    assert gas_full_error < .001
    assert fixing_error < .001

    gas_summary = average(grows, ['rho','variable'], ['GAS'])
    check_rows, conditional_rows = [], []
    if not args.pilot:
        for rho in [0., .4, .7]:
            for rep in range(3):
                C, _, _ = gas_matrix(15, 10310917+rep*1009, rho, ORDERS[0], 'normal', 50)
                eigen, vectors = np.linalg.eigh(C)
                scores, _, _ = GAS_score_95(vectors, eigen)
                for i, score in enumerate(scores):
                    check_rows.append(dict(rep=rep,rho=rho,variable=i+1,GAS=float(score)))
                cond = conditional_total(18, 11310917+rep*1009, rho)
                for i, value in enumerate(cond):
                    conditional_rows.append(dict(rep=rep,rho=rho,variable=i+1,ST_ind=float(value)))
            print(f'Validation complete at rho={rho:.1f}, elapsed={time.time()-started:.1f}s', flush=True)
        save_csv(OUT/'gas_convergence_replicates.csv', check_rows)
        save_csv(OUT/'conditional_check_replicates.csv', conditional_rows)

    high = average(check_rows, ['rho','variable'], ['GAS']) if check_rows else []
    conditional = average(conditional_rows, ['rho','variable'], ['ST_ind']) if conditional_rows else []
    gas_diff = max((abs(row['GAS']-next(r['GAS'] for r in gas_summary
         if r['rho']==row['rho'] and r['variable']==row['variable'])) for row in high), default=0.)
    conditional_diff = max((abs(row['ST_ind']-next(r['ST'] for r in sobol_summary
         if r['rho']==row['rho'] and r['variable']==row['variable'] and r['family']=='ind'))
         for row in conditional), default=0.)
    assert gas_diff < .002
    assert conditional_diff < .003
    audit = dict(sobol_independence_max_abs_error=float(sobol_error),
                 full_GAS_independence_max_abs_error=gas_full_error,
                 fixing_independence_max_abs_error=float(fixing_error),
                 gas_refinement_max_abs_change=float(gas_diff),
                 direct_conditional_total_max_abs_difference=float(conditional_diff))
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    metadata = dict(fresh_run=True, pilot=args.pilot, model_a=A.tolist(), rhos=rhos,
                    gas_coordinate='independent standard normal drivers', gas_generation_order=[1,2,3,4],
                    gas_N=2**gas_power, quadrature_nodes=quad, sobol_N=2**sobol_power,
                    fixing_N=2**fix_power, repeats=repeats,
                    seeds={k:[v+1009*r for r in range(repeats)] for k,v in SEEDS.items()},
                    fixing_values=[.5,.25], fixing_protocol='Physical substitution; remaining joint marginal unchanged',
                    gas_selection_fraction=.95, elapsed_seconds=time.time()-started,
                    numpy_version=np.__version__)
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Validation:', json.dumps(audit), flush=True)
    print(f'DONE in {time.time()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
