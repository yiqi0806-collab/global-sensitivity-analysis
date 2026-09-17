"""Fresh Sobol estimates and a common holdout for GAS/Sobol variable selection."""
from pathlib import Path
import csv
import itertools
import json
import time
from sergei_g_function_study import (
    ROOT, A, ORDERS, np, qmc, ndtri, transform, model, sobol_four, save_csv, average,
)
from sergei_g_function_fixing_study import conditional_total

OUT=ROOT/'source/results/sergei_sobol_gas_comparison'
BASELINE=ROOT/'source/results/sergei_g_function_fixing_rerun'
RHOS=np.round(np.arange(-.7,.701,.1),2).tolist()
REPEATS=6
SOBOL_SEED=14310971
HOLDOUT_SEED=15310987


def read_numeric(path):
    with path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key not in ['method','family','fixed','selected']:
                row[key]=float(row[key])
    return rows


def all_fixing(power, seed, rho):
    x=transform(ndtri(qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)),rho,ORDERS[0])
    factors=(np.abs(4*x-2)+A)/(1+A)
    y=np.prod(factors,axis=1)
    variance=y.var()
    rows=[]
    for count in [1,2]:
        for subset in itertools.combinations(range(4),count):
            remaining=[i for i in range(4) if i not in subset]
            rest=np.prod(factors[:,remaining],axis=1)
            for c in [.5,.25]:
                factor=np.prod((abs(4*c-2)+A[list(subset)])/(1+A[list(subset)]))
                yr=rest*factor
                # Check the factorized evaluation against direct physical substitution.
                xx=x[:64].copy(); xx[:,subset]=c
                assert np.allclose(yr[:64],model(xx),atol=1e-13,rtol=1e-13)
                mse=np.mean((y-yr)**2)
                rows.append(dict(rho=rho,count=count,fixed='+'.join('x'+str(i+1) for i in subset),
                                 value=c,rmse_over_sd=float(np.sqrt(mse/variance)),
                                 mse_over_variance=float(mse/variance)))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.time()
    gas=read_numeric(BASELINE/'gas_summary.csv')
    gas_meta=json.loads((BASELINE/'run_metadata.json').read_text())
    assert gas_meta['repeats']==6 and gas_meta['gas_generation_order']==[1,2,3,4]
    save_csv(OUT/'gas_baseline_summary.csv',gas)
    srows,hrows=[],[]
    for rep in range(REPEATS):
        for rho in RHOS:
            for row in sobol_four(19,SOBOL_SEED+1009*rep,rho):
                srows.append(dict(rep=rep,**row))
            for row in all_fixing(19,HOLDOUT_SEED+1009*rep,rho):
                hrows.append(dict(rep=rep,**row))
            print(f'Repeat {rep+1}/6, rho={rho:+.1f}, elapsed={time.time()-started:.1f}s',flush=True)
        save_csv(OUT/'sobol_replicates.csv',srows)
        save_csv(OUT/'holdout_all_subsets_replicates.csv',hrows)
        save_csv(OUT/'sobol_summary.csv',average(srows,['rho','variable','family'],['S','ST']))
        save_csv(OUT/'holdout_all_subsets_summary.csv',average(hrows,['rho','count','fixed','value'],['rmse_over_sd','mse_over_variance']))

    sobol=average(srows,['rho','variable','family'],['S','ST'])
    holdout=average(hrows,['rho','count','fixed','value'],['rmse_over_sd','mse_over_variance'])
    selections, paired=[],[]
    for rho in RHOS:
        scores={
            'GAS':{int(r['variable']):r['GAS'] for r in gas if r['rho']==rho},
            'Full total Sobol':{int(r['variable']):r['ST'] for r in sobol if r['rho']==rho and r['family']=='full'},
            'Independent total Sobol':{int(r['variable']):r['ST'] for r in sobol if r['rho']==rho and r['family']=='ind'},
        }
        for count in [1,2]:
            for method,values in scores.items():
                chosen=sorted(sorted(values,key=lambda i:(values[i],i))[:count])
                name='+'.join('x'+str(i) for i in chosen)
                for c in [.5,.25]:
                    row=next(r for r in holdout if r['rho']==rho and r['count']==count and r['fixed']==name and r['value']==c)
                    best=min(r['rmse_over_sd'] for r in holdout if r['rho']==rho and r['count']==count and r['value']==c)
                    selections.append(dict(rho=rho,method=method,count=count,value=c,selected=name,
                                           rmse_over_sd=row['rmse_over_sd'],rmse_over_sd_se=row['rmse_over_sd_se'],
                                           best_observed_rmse=best,excess_over_best=row['rmse_over_sd']-best))
                    for rep in range(REPEATS):
                        hr=next(r for r in hrows if r['rep']==rep and r['rho']==rho and r['fixed']==name and r['value']==c)
                        paired.append(dict(rep=rep,rho=rho,method=method,count=count,value=c,selected=name,
                                           rmse_over_sd=hr['rmse_over_sd']))
    save_csv(OUT/'selected_fixing_summary.csv',selections)
    save_csv(OUT/'selected_fixing_replicates.csv',paired)

    q=1/(3*(1+A)**2); variance=np.prod(1+q)-1
    exactS=q/variance; exactST=q*np.prod(1+q)/(1+q)/variance
    independent_error=max(abs(r[key]-exact[int(r['variable'])-1]) for r in sobol if r['rho']==0
                          for key,exact in [('S',exactS),('ST',exactST)])
    assert independent_error<.001
    fix_error=0.
    for row in holdout:
        if row['rho']!=0: continue
        subset=[int(n[1:])-1 for n in row['fixed'].split('+')]
        remaining=[i for i in range(4) if i not in subset]
        k=np.prod((abs(4*row['value']-2)+A[subset])/(1+A[subset]))
        mse=np.prod(1+q[remaining])*(np.prod(1+q[subset])-2*k+k*k)
        exact=np.sqrt(max(0.,mse)/variance)
        fix_error=max(fix_error,abs(row['rmse_over_sd']-exact))
    assert fix_error<.001

    high,conditional=[],[]
    for rho in [0.,.4,.7]:
        for rep in range(3):
            for row in sobol_four(20,16311003+1009*rep,rho):
                high.append(dict(rep=rep,**row))
            for i,value in enumerate(conditional_total(19,17311017+1009*rep,rho),1):
                conditional.append(dict(rep=rep,rho=rho,variable=i,ST_ind=float(value)))
        print(f'Validation complete rho={rho:.1f}, elapsed={time.time()-started:.1f}s',flush=True)
    save_csv(OUT/'sobol_refinement_replicates.csv',high)
    save_csv(OUT/'conditional_validation_replicates.csv',conditional)
    high_summary=average(high,['rho','variable','family'],['S','ST'])
    total_diff=max(abs(r['ST']-next(t['ST'] for t in sobol if all(t[k]==r[k] for k in ['rho','variable','family']))) for r in high_summary)
    conditional_summary=average(conditional,['rho','variable'],['ST_ind'])
    cond_diff=max(abs(r['ST_ind']-next(t['ST'] for t in sobol if t['family']=='ind' and t['rho']==r['rho'] and t['variable']==r['variable'])) for r in conditional_summary)
    assert total_diff<.001 and cond_diff<.001
    two={r['selected'] for r in selections if r['count']==2}
    # Identical subsets must use exactly the same held-out evaluations.
    for rho in RHOS:
        for c in [.5,.25]:
            rows=[r for r in selections if r['rho']==rho and r['value']==c and r['count']==2]
            if len({r['selected'] for r in rows})==1:
                assert len({r['rmse_over_sd'] for r in rows})==1
    audit=dict(sobol_independence_max_abs_error=float(independent_error),
               holdout_independence_max_abs_error=float(fix_error),
               sobol_total_refinement_max_abs_change=float(total_diff),
               conditional_ST_ind_max_abs_difference=float(cond_diff),
               two_variable_selected_sets=sorted(two),
               two_variable_setting_count=len([r for r in selections if r['count']==2]),
               maximum_two_variable_excess_over_best=max(r['excess_over_best'] for r in selections if r['count']==2))
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    metadata=dict(model_a=A.tolist(),rhos=RHOS,repeats=REPEATS,sobol_N=2**19,holdout_N=2**19,
                  sobol_seeds=[SOBOL_SEED+1009*r for r in range(REPEATS)],
                  holdout_seeds=[HOLDOUT_SEED+1009*r for r in range(REPEATS)],
                  gas_source=str(BASELINE.relative_to(ROOT)),gas_recomputed_this_run=False,
                  gas_metadata=gas_meta,selection='One or two smallest scores; no shared numerical cutoff',
                  holding_protocol='Same independent evaluation sample for every method; physical substitution',
                  fixing_values=[.5,.25],all_tested_subsets='all four singletons and six pairs',
                  elapsed_seconds=time.time()-started)
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
    print('rho=.7 decisions:',json.dumps([r for r in selections if r['rho']==.7]),flush=True)


if __name__=='__main__': main()
