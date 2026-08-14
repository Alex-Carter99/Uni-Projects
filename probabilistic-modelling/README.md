# Probabilistic Modelling

Aston University MSc Data Science · 2024–25  
Two assessed courseworks in Bayesian inference, graphical models, and mixture modelling.

## Contents

| File | What it is |
| --- | --- |
| `coursework-1-brief.pdf` | Official Coursework 1 brief |
| `coursework-1-submission.pdf` | My Coursework 1 report (code in the appendix) |
| `coursework-2-brief.pdf` | Official Coursework 2 brief |
| `coursework-2-submission.pdf` | My Coursework 2 report (code in the appendix) |

## Coursework 1

- **Poisson–Gamma inference** on 10,000 hospital birth counts: histogram and prior, maximum likelihood λ_ML = 4.968, posterior mean/mode, MAP λ_MAP = 4.9676, and a 95% credible interval (4.924, 5.011).
- **Two-class density model** for weekly homework hours (failed students ~ Uniform, passed ~ Normal), unknown-label classification, decision boundary, and a 37.76% misclassification rate for fail-as-pass errors.
- **Bayesian network** for a disease-transmission story (Ruth → James, symptoms, noisy test). Directed graph, joint factorisation in fractions, and exact posterior updates as evidence changed.

## Coursework 2

- **Coupled Poisson mixture:** PMF, expectation and variance of an equal mixture of two Poisson components.
- **Dealership similarity:** normalised daily sales for London, New York, Doha and Paris; location-specific maximum-likelihood fits (same family, different parameters).
- **Gaussian mixture / EM** on Old Faithful eruptions (150-point student-number slice). Trained K = {2, 3, 4, 5}, plotted the mixtures and log-likelihood paths, and selected **K = 2**. Implemented EM from scratch and verified it against scikit-learn on the same data.
