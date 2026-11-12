# Experiment 9: Stochastic GAS Estimands in a Real PyTorch Network

## Objective

Experiment 8 showed analytically that a stochastic Global Activity Subspace
(GAS) calculation can target two different matrices. A one-realization CRN
calculation estimates the pathwise matrix

\[
C_{\mathrm{path}}=\mathbb{E}[QQ^\top],
\]

whereas sensitivity of the expected network response is represented by

\[
C_\mu=\mathbb{E}\{\mathbb{E}[Q\mid X,V]\mathbb{E}[Q\mid X,V]^\top\}.
\]

This experiment tests whether the same distinction is visible in a trained
PyTorch neural network on real data. The high-budget matrices used here are
numerical references, not analytic ground truth.

## Data and neural network

The study uses the scikit-learn diabetes regression dataset. It contains 442
patients, 10 standardized clinical inputs, and a continuous disease-progression
response. The data are divided into training, validation, sensitivity, and
held-out test sets. Standardization is fitted using the training set only.

Each independently trained model is a `10-64-32-1` ReLU network. The two hidden
layers contain 64 and 32 neurons. A 20% dropout rate is applied after each
hidden layer. Training uses Adam, mini-batches of 32, and early stopping based
on the independent validation set. The best validation checkpoint is frozen
before any GAS calculation. Five training seeds are treated as the independent
statistical units.

The five networks obtain held-out test \(R^2\) values from 0.458 to 0.507, with
a mean of 0.484. Thus the fitted models have useful, although not perfect,
predictive skill. The purpose is to compare stochastic sensitivity estimators
for the same fitted network rather than to claim state-of-the-art prediction on
this dataset.

## Explicit dropout and CRN

Ordinary PyTorch dropout draws masks internally. This implementation instead
supplies a separate mask for every observation and hidden layer. For each
coordinate replacement, the baseline and perturbed input use exactly the same
two masks. This is the common-random-numbers (CRN) comparison. The negative
control gives the baseline and perturbed input unrelated masks.
The negative control uses a separate baseline mask for each coordinate and
therefore costs \(2Nd\) forward evaluations, versus \(N(d+1)\) for one-mask
CRN. It is a protocol control, not a compute-matched competitor.

The coordinate-replacement secant for input \(j\) is

\[
Q_j=\frac{F(x^{(j)},\xi)-F(x,\xi)}{v_j-x_j},
\]

where \(x^{(j)}\) replaces only coordinate \(j\), and \(\xi\) denotes the two
dropout masks. Replacement values are sampled from the training-set empirical
marginals with a minimum standardized separation of 0.10. This is a global
marginal-replacement sensitivity design, not a causal intervention.

## Compared estimators

Four protocols are evaluated on the same anchors and replacements:

1. **One-mask CRN:** one paired dropout realization per anchor, targeting
   \(C_{\mathrm{path}}\).
2. **K-mask average:** average \(K\) secant vectors and then form an outer
   product. Its finite-\(K\) target is approximately
   \(C_\mu+(C_{\mathrm{path}}-C_\mu)/K\).
3. **Cross-mask U-statistic:** average cross-products from different masks,
   directly targeting \(C_\mu\).
4. **Independent masks:** use unrelated masks for the baseline and replacement;
   this is a negative control.

The full run uses 96 sensitivity anchors, \(K\in\{1,2,4,8,16\}\), 128 masks
for each numerical reference, 12 internal Monte Carlo repeats, and five
independently trained networks. Internal repeats are averaged within a network;
confidence intervals are calculated across the five networks.

## Results

### Pairing is essential for the pathwise matrix

The one-mask paired estimator has mean relative Frobenius error 0.055 with
respect to its numerical \(C_{\mathrm{path}}\) reference. The independent-mask
control has error 4.065. The paired, cross-network difference
`independent minus paired` is 4.010, with a 95% Student-t interval of
[3.526, 4.495].

The paired estimator also recovers the leading reference axis with mean
absolute cosine 0.9996, compared with 0.6165 for independent masks. The paired
advantage is 0.3831, with a 95% interval of [0.3121, 0.4540]. Independent masks
inflate the estimated 90%-energy dimension to 7.5, whereas the numerical
pathwise target has mean dimension 2.4.

### Averaging approaches the mean-response matrix

The relative error of the K-mask average to numerical \(C_\mu\) decreases as
the number of masks grows:

| K | K-mask average | Cross-mask U-statistic |
|---:|---:|---:|
| 1 | 0.0696 | -- |
| 2 | 0.0458 | 0.0424 |
| 4 | 0.0297 | 0.0283 |
| 8 | 0.0209 | 0.0205 |
| 16 | 0.0141 | 0.0140 |

For the K-mask estimator, the K=1 to K=16 error reduction is 0.0555, with a
95% interval of [0.0461, 0.0649]. At K=2, the U-statistic has a small observed
advantage of 0.00348, with an unadjusted 95% interval of [0.00027, 0.00670].
At larger K, the observed differences are small and their intervals include
zero. These K-specific comparisons were not adjusted for multiple testing, so
the K=2 result is descriptive rather than a general superiority claim. The
overall pattern is consistent with the two estimators approaching the same
mean-response target as stochastic averaging increases.

## Conclusion

The real-data PyTorch results reproduce the central conclusion of the analytic
study. A one-mask CRN calculation is a stable estimator of pathwise stochastic
sensitivity, but it should not be described as sensitivity of the expected
network response. Averaging multiple paired masks moves the estimate toward
the mean-response matrix, while the cross-mask U-statistic targets that matrix
directly and showed its clearest observed advantage at small K in this
experiment. Independent baseline and perturbation masks are not a harmless
implementation choice; they create a large false sensitivity component and an
artificially high-dimensional GAS.

This is a successful regression-stage transfer, not yet a general neural-network
result. The numerical references still contain finite Monte Carlo error, the
diabetes mean-response GAS is strongly low-dimensional, and only one real
dataset, one fixed data split, one fixed anchor/replacement design, and one
architecture have been studied. The Student-t intervals quantify variation
across training randomness conditional on this dataset, split, and sensitivity
design; they do not quantify dataset-sampling uncertainty.
The next experiment should add a real classification dataset, use logits rather
than probabilities, and then repeat the analysis across architectures and
dropout rates.
