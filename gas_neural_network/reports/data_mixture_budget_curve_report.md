# GAS Direction-Estimation Budget Curve for Neural-Network Data Mixtures

## Abstract

This experiment tests whether the inconclusive optimization result in the
initial six-source loss-reweighting pilot was associated with an inadequate GAS
direction-estimation budget. The data sources, simplex parameterization,
one-hidden-layer ReLU model, training schedule, validation objective, search
radius, and 20-candidate search budget are held fixed. Only the number of
nested outer GAS design points changes. Common-random-number (CRN) GAS is
compared with an independent-randomness protocol over six paired repeats.

Increasing the CRN-GAS budget makes the normalized activity matrix more similar
to a fixed high-budget CRN reference. The leading-direction absolute cosine
rises from 0.853 at 42 logical train-and-evaluate calls to 0.992 at 504 calls,
whereas the independent-randomness estimate remains unstable. Conditional on a
fixed 20-candidate search budget, CRN-GAS first exceeds random search with a
positive unadjusted 95% paired interval at 84 calls and exceeds coordinate search at 168
calls. These are promising conditional results, but they are not yet an
end-to-end compute comparison or evidence of generalization beyond the fixed
synthetic data problem.

## Experimental design

The optimization variables are the weights of six synthetic training-data
sources: three relatively clean domain sources and three degraded sources.
Weights lie on the simplex and are represented by five orthonormal centered
log-ratio coordinates. Each training step draws eight observations from every
source and scales their losses by the source weights. Thus this is more
precisely a source-loss-reweighting study than a literal data-mixture sampler.
Each objective evaluation retrains the same ReLU MLP for 120 steps and returns
balanced clean-validation mean squared error.

The nested outer-sample sizes are $m\in\{2,4,8,16,24\}$. With two split
Gauss--Legendre nodes per coordinate, one standalone GAS estimate has the
logical cost

\[
  m\{1+2qd\}=m\{1+2(2)(5)\}=21m,
\]

giving budgets of 42, 84, 168, 336, and 504 train-and-evaluate calls. Larger
designs contain every point from smaller designs. A separate CRN estimate with
$m=24$ and $q=3$, costing 744 calls, is used as a pathwise reference; it is
not treated as ground truth.

For CRN-GAS, the baseline and all coordinate perturbations at an outer point
reuse the same initialization and minibatch stream. The independent protocol
redraws these random quantities. Each estimated CRN basis is then used to
construct 20 paired candidate mixtures. Candidate selection averages two
training seeds, and the selected mixture is evaluated on eight disjoint
training seeds. Random, coordinate, and fixed high-budget GAS-reference
candidate pools are held identical across the five GAS budgets within a
repeat. Every method uses five paired directions at the full search radius and
five at half radius, so direction quality is not confounded with a different
step-size schedule.

## Direction-estimation results

| Logical calls | CRN leading cosine | Independent leading cosine | CRN normalized matrix error | CRN 90% dimension |
|---:|---:|---:|---:|---:|
| 42  | 0.853 | 0.126 | 0.446 | 2.50 |
| 84  | 0.915 | 0.281 | 0.360 | 3.33 |
| 168 | 0.971 | 0.355 | 0.287 | 3.50 |
| 336 | 0.980 | 0.211 | 0.281 | 3.50 |
| 504 | 0.992 | 0.193 | 0.272 | 3.67 |

The trace-normalized Frobenius error falls by 0.174 between 42 and 504 calls,
with a paired 95% Student-t interval of [0.103, 0.245]. Thus the full CRN
activity estimate improves materially with budget. The corresponding increase
in leading-direction cosine is 0.139, but its interval [-0.046, 0.324] includes
zero because the lowest-budget cosine is already high in several repeats and
only six repeats are available.

Within this finite-difference construction, CRN is clearly more stable than
the naive independent-seed control for leading-direction recovery at every
tested budget. At 504 calls, the paired CRN-minus-independent
cosine difference is 0.799 [0.389, 1.210]. The unbounded upper endpoint is a
property of the small-sample Student-t interval; the observed cosine itself is
bounded by one.

The reference activity spectrum explains 52.5%, 20.6%, 14.8%, 8.7%, and 3.4%
of activity. Four of five directions are required for 90% energy. The problem
therefore remains nearly full-dimensional; the benefit below is better
orientation and ordering of search directions, not a dramatic reduction to a
one-dimensional problem.

## Mixture-search results

| Logical GAS calls | CRN-GAS improvement | CRN-GAS minus random | 95% interval | CRN-GAS minus coordinate | 95% interval |
|---:|---:|---:|:---:|---:|:---:|
| 42  | 0.02594 | 0.00715 | [-0.00351, 0.01782] | -0.00276 | [-0.01130, 0.00577] |
| 84  | 0.03027 | 0.01148 | [0.00264, 0.02032] | 0.00157 | [-0.00578, 0.00891] |
| 168 | 0.03228 | 0.01349 | [0.01001, 0.01697] | 0.00357 | [0.00015, 0.00700] |
| 336 | 0.03300 | 0.01421 | [0.01128, 0.01713] | 0.00429 | [0.00266, 0.00593] |
| 504 | 0.03248 | 0.01369 | [0.01096, 0.01642] | 0.00378 | [0.00288, 0.00468] |

The fixed baselines improve validation MSE by 0.01879 for random search,
0.02870 for coordinate search, and 0.03206 for the high-budget GAS reference.
From 168 calls onward, no difference from the fixed GAS reference is detected,
and unadjusted paired intervals favor CRN-GAS over both random and coordinate
search. Failure to detect a reference difference is not an equivalence result.
Mean improvement plateaus near 168 calls and is not strictly monotone with
budget.

The initial pilot used one low-budget direction estimate for every search
repeat and obtained a null comparison against random and coordinate search.
The present design propagates direction-estimation uncertainty by pairing each
search repeat with its own nested GAS estimate. The earlier null result does
not persist under this revised protocol. This is not a strict replication,
because the candidate-generation, search, and verification seeds also differ.

## What this exploratory study supports

1. Within this boundary-sensitive finite-difference design, CRN-GAS is much
   more stable than the deliberately naive independent-seed control. Simply
   increasing that control's budget does not recover a stable direction.
2. Additional CRN budget improves the full activity-matrix estimate, with most
   leading-direction recovery already achieved by approximately 168 calls.
3. Under a fixed 20-candidate search budget and matched radius schedule, the
   unadjusted paired intervals favor CRN-GAS over random and coordinate
   directions from 168 calls onward on this fixed synthetic problem.
4. GAS is useful here mainly as a rotated and prioritized search basis. The
   reference spectrum does not support a strong low-dimensionality claim.

## Limitations

- Direction-estimation cost is reported separately from the 20-candidate
  search budget. The result is conditional direction quality, not an
  end-to-end compute advantage.
- The statistical unit is one random-design/training/search repeat conditional
  on a single generated dataset. Six repeats do not support population-level
  generalization.
- The 744-call reference is a fixed high-budget CRN estimate rather than an
  analytic truth.
- Several budgets and contrasts are inspected without multiplicity correction;
  threshold claims should be confirmed on held-out seeds.
- Independent training noise is divided by coordinate displacements that can
  become small near the boundary. Its resulting heavy-tailed error shows that
  this naive control fails; it does not prove that every independent-noise
  estimator must fail.
- The adaptive 90%-energy dimension changes with budget, so the search curve
  evaluates the complete estimated-basis procedure rather than orientation
  alone.
- The current weighted-risk minibatch uses examples from every source at every
  step. A real data-mixture sampler should instead draw source identities
  according to the mixture weights.

## Recommended next stage

Use a development/confirmation split. Freeze one GAS budget before looking at
confirmation results, repeat the comparison across independently generated
datasets, and charge direction estimation plus search in one end-to-end
training budget. A strong follow-up benchmark should use 12 raw sources from a
predeclared $3\times2\times2$ domain-by-quality-by-difficulty design. GAS
should still operate on the 11-dimensional raw simplex; a four-dimensional
semantic contrast space should be used only for post-hoc alignment tests with
a label-permutation null. This avoids manufacturing a low-dimensional success
by defining only two or three semantic optimization variables.

## Reproducibility

Run from `gas_neural_network`:

```bash
python experiments/gas_data_mixture_budget_curve.py
```

The script writes raw repeat-level records, aggregate summaries, paired
contrasts, metadata, and the figure to
`results/data_mixture_budget_curve/`.
