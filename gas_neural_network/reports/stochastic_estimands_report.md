# Experiment 8: Analytic Validation of Stochastic GAS Estimands

## Objective

This experiment determines which Global Activity Subspace (GAS) matrix is
estimated when the black-box response is stochastic. It distinguishes the GAS
of the expected response,

\[
C_\mu
=
\mathbb E_{X,V}
\left[
\mathbb E_\xi(Q^\xi\mid X,V)
\mathbb E_\xi(Q^\xi\mid X,V)^\top
\right],
\]

from the expected GAS of individual stochastic paths,

\[
C_{\mathrm{path}}
=
\mathbb E_{X,V,\xi}[Q^\xi(Q^\xi)^\top]
=C_\mu+C_{\mathrm{rand}}.
\]

The two matrices answer different questions. The first describes how the mean
prediction or loss changes with the inputs. The second also includes changes in
the slopes of individual random realizations.

## Analytic benchmark

The response is

\[
F(x,\xi)
=
\sum_{i=1}^{6}\alpha_i x_i^2
+\tau\xi b^\top x,
\qquad \xi\sim N(0,1),
\]

where

\[
\alpha=(3,2,1,0.5,0.25,0.125),
\qquad
b=\frac{1}{\sqrt 2}(0,0,0,1,1,0)^\top.
\]

The anchor and replacement coordinates are sampled on \([-1,1]\). With no
minimum replacement separation, the exact matrices are

\[
C_\mu=\frac{2}{3}\operatorname{diag}(\alpha_1^2,\ldots,\alpha_6^2),
\qquad
C_{\mathrm{rand}}=\tau^2bb^\top.
\]

At \(\tau=3\), the leading eigenvalue of \(C_\mu\) is 6 and its leading axis is
\(e_1\). The leading eigenvalue of \(C_{\mathrm{path}}\) is 9.105 and its
leading axis is close to \((e_4+e_5)/\sqrt2\). The benchmark therefore makes
the two scientific targets visibly different.

## Estimators

| Protocol | Matrix formed at each outer anchor | Intended target |
|---|---|---|
| One-seed CRN | \(Q_1Q_1^\top\) | \(C_{\mathrm{path}}\) |
| \(K\)-seed average | \(\bar Q_K\bar Q_K^\top\) | \(C_\mu+C_{\mathrm{rand}}/K\) |
| Cross-seed U-statistic | \(\{K(K-1)\}^{-1}\sum_{k\ne\ell}Q_kQ_\ell^\top\) | \(C_\mu\) |
| Independent states | Independent random states in the baseline and perturbed evaluations | Negative protocol control |

For every CRN secant vector, the baseline and all six coordinate replacements
share the same random state. Different outer anchors use independent random
states.

## Experimental design

- 200 independent Monte Carlo repetitions;
- \(K\in\{1,2,4,8,16,32\}\);
- outer sample sizes \(N\in\{64,128,256,512,1024\}\);
- stochastic amplitudes \(\tau\in\{0,1,2.5,3\}\);
- minimum replacement separations
  \(h\in\{0,0.01,0.02,0.05,0.10,0.20\}\);
- one fixed-budget comparison with \(NK=4096\) coupled secant vectors.

For the separation experiment, \((X_i,V_i)\) is jointly resampled until
\(|V_i-X_i|\ge h\). The exact analytic matrix is adjusted for this conditional
design. Reported logical evaluation costs are the costs of running each
estimator separately; the implementation reuses nested \(K\) and \(N\) samples
to reduce wall-clock time.

## Results

### One-seed CRN estimates pathwise activity

At \(\tau=3\) and \(N=256\), the mean one-seed matrix has relative bias 0.0077
with respect to \(C_{\mathrm{path}}\), and its leading-axis cosine with the
pathwise axis is 0.99999. In contrast, its relative bias with respect to
\(C_\mu\) is 1.3737 and its leading-axis cosine with the mean-response axis is
0.0041. One-seed CRN is therefore not a noisy estimator of \(C_\mu\); it is an
accurate estimator of a different matrix.

### The K-average bias follows the predicted 1/K law

At \(\tau=3\) and \(N=256\):

| \(K\) | Empirical relative bias to \(C_\mu\) | Exact relative bias |
|---:|---:|---:|
| 1 | 1.3737 | 1.3632 |
| 2 | 0.6888 | 0.6816 |
| 4 | 0.3416 | 0.3408 |
| 8 | 0.1707 | 0.1704 |
| 16 | 0.0856 | 0.0852 |
| 32 | 0.0435 | 0.0426 |

The empirical curve agrees closely with
\(\|C_{\mathrm{rand}}\|_F/(K\|C_\mu\|_F)\).

### The cross-seed U-statistic directly targets the mean response

The mean U-statistic matrix has relative bias between 0.007 and 0.009 across
\(K=2,\ldots,32\), consistent with finite Monte Carlo error rather than a
systematic bias. Its mean leading-axis cosine with \(C_\mu\) is approximately
0.99998.

At fixed \(N=256\), its mean relative matrix error decreases from 0.1528 at
\(K=2\) to 0.0955 at \(K=32\). The improvement eventually plateaus because
increasing the number of random states cannot remove variability across outer
design points.

### All valid estimators converge to their own targets

Increasing \(N\) from 64 to 1024 reduces intended-target error as follows:

| Protocol | \(N=64\) | \(N=1024\) |
|---|---:|---:|
| One-seed CRN to \(C_{\mathrm{path}}\) | 0.2070 | 0.0549 |
| \(K=8\) average to \(C_\mu+C_{\mathrm{rand}}/8\) | 0.1895 | 0.0481 |
| \(K=8\) U-statistic to \(C_\mu\) | 0.1931 | 0.0491 |

A sixteen-fold increase in outer sample size produces approximately a
four-fold reduction in error, consistent with \(N^{-1/2}\) convergence.

### More inner random states are not always better at fixed cost

With \(NK=4096\) coupled secant vectors, the U-statistic has its smallest
relative MSE at \(K=2\), equal to 0.00304. Its MSE increases to 0.01991 at
\(K=32\), where only 128 outer anchors remain. The biased \(K\)-average reaches
its best MSE at \(K=16\), equal to 0.01770, and then worsens at \(K=32\).
This demonstrates the tradeoff between inner stochastic averaging and outer
input-space coverage.

### Independent random states are an invalid negative control

The independent-state secant divides an unrelated stochastic output difference
by the replacement distance. Its median relative error to \(C_\mu\) is 7063
when no minimum separation is imposed, with a 90th percentile of 233111. The
median remains 234.0 at \(h=0.01\) and 11.57 even at \(h=0.20\). By comparison,
one-seed CRN has median relative error of approximately 0.10 with respect to its
correct \(C_{\mathrm{path}}\) target across the separation grid.

### Raw U-statistic matrices can be indefinite

The fraction of raw U-statistic estimates with a negative eigenvalue is about
47%, 48%, 36%, 16%, and 1% for \(K=2,4,8,16,32\), respectively. Positive
semidefinite projection slightly lowers average finite-sample error, but it
changes the unbiased estimator and must be reported separately.

## Conclusion

The experiment confirms the stochastic GAS estimand distinction on a problem
with exact ground truth. One-seed CRN-GAS estimates pathwise activity,
\(K\)-seed averaging approaches mean-response activity with the predicted
\(1/K\) bias, and the cross-seed U-statistic directly estimates mean-response
activity. Independent baseline and perturbation states do not estimate either
target reliably because random output changes are amplified by small
replacement distances.

## Limitations and next step

This is an analytic estimator-validation experiment, not yet a neural-network
or real-data result. The stochastic component is scalar Gaussian and rank one,
and the benchmark is deliberately designed to separate the two leading axes.
The next experiment should implement the pathwise and mean-response estimators
for Monte Carlo dropout in PyTorch and test them on real regression and
classification datasets across independently trained models.
