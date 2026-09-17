# GAS and Sobol comparison for the correlated g-function

Calculation code for Global Activity Scores (GAS), generalized Sobol indices,
and the error caused by fixing one or two physical inputs of a four-variable
g-function.

## Run

Use Python 3.10 or newer. From the repository root:

```bash
cd g_function_comparison
python -m pip install -r requirements.txt
python code/sergei_g_function_fixing_study.py
python code/compare_sobol_gas_fixing.py
```

Run the commands in this order. The first calculation creates the GAS baseline;
the second recomputes Sobol indices and compares variable-fixing decisions using
that baseline. Both calculations write their results under `source/results/`.
Subsequent runs replace those locally generated files. The full experiments
may take several minutes and use arrays with more than 500,000 samples.

## Files

- `code/sergei_g_function_study.py`: the g-function, Gaussian copula, GAS
  divided-difference matrix and generalized Sobol estimators.
- `code/gas_energy.py`: GAS scores from the retained eigenpairs.
- `code/sergei_g_function_fixing_study.py`: GAS baseline and fixing study.
- `code/compare_sobol_gas_fixing.py`: Sobol comparison, independent evaluation
  samples, subset selection and numerical validation.

The model uses uniform marginals, g-function parameters `(0, 1, 99, 99)`, and
a sweep of Gaussian copula correlations from -0.7 to 0.7. GAS uses the independent
normal-driver construction 1 -> 2 -> 3 -> 4.

Four Sobol families are calculated: full first-order, full total-order,
independent first-order and independent total-order. For variable selection,
GAS is compared with the two total-order families. Each method chooses one
or two inputs with the smallest scores, then fixes them at 0.5 or 0.25.

The error is RMSE divided by the standard deviation of the original output.
Multiply `rmse_over_sd` by 100 to express it as a percentage. All methods share
the same independent evaluation samples, so identical choices have identical
fixing errors. The scripts include analytical checks at independence, sampling
refinement, and separate conditional resampling for independent total effects.

中文：先安装依赖，再按上述顺序运行两个脚本。第一个脚本生成 GAS 基线，
第二个脚本计算 Sobol 并比较固定变量后的 RMSE。计算结果保存在本地
`source/results/` 目录。CSV 中的 `rmse_over_sd` 乘以 100 后才是百分数。
