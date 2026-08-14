# AM41PR Research Project

Aston University MSc Data Science · 60-credit dissertation · September 2025  
**Visa Compliance for the Future: A Machine Learning Perspective**  
Supervisor: Matthew Stuart Bickley

An end-to-end supervised learning pipeline for institutional visa-sponsorship risk: classifying credibility-interview outcomes on anonymised CAS-shield applications, so high-risk cases can be flagged before interview.

## Contents

| File | What it is |
| --- | --- |
| `dissertation.pdf` | Full submitted dissertation (10,116 words) |
| `project-guidelines.pdf` | Official AM41PR project guidelines |
| `final-pipeline.ipynb` | Reproducible modelling notebook |
| `Combined.csv` | Anonymised operational dataset used in the study |

## Problem

UK higher-education sponsors face tighter compliance duties after the 2025 immigration white paper. Credibility interviews are expensive and uneven; the operational question is whether a model trained on historical CAS-shield records can surface likely failures early enough to change how interviews are allocated.

## What I did

- Assembled and cleaned **3,957 anonymised applications** across **16 features** and **four intake cycles (2022–2025)**.
- Ran a full supervised workflow: preprocessing, exploratory analysis, classifier comparison, then two-stage hyperparameter search (RandomizedSearchCV followed by GridSearchCV).
- Selected **Random Forest** as the production candidate after that comparison.
- Evaluated on a held-out test set and with **5-fold cross-validation**, with particular attention to the minority **Failed** class — the outcome that matters for compliance risk.
- Interpreted the fitted model: **accommodation, age, student deposit and course fee** accounted for roughly **94%** of feature importance, so the ranking is something a compliance officer can interrogate rather than a black box.

## Results

| Metric | Result |
| --- | --- |
| Test balanced accuracy | **0.8456** (+0.0472 vs baseline) |
| ROC-AUC | **0.8876** |
| Minority “Failed” recall | **88.89%** |
| 5-fold CV mean balanced accuracy | **0.8614 ± 0.0271** |

The work is framed as a **proof of concept** for institutional risk screening, with ethics, class imbalance and operational limitations discussed in the dissertation — not as a claim that the model replaces human judgement.
