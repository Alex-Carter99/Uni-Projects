# AM41DP Data Science Programming

Aston University MSc Data Science · 2024–25  
**Grade: 91.1%**

Coursework for AM41DP (Data Science Programming). Two tasks: a constraint-propagation solver in Python, and a full exploratory / predictive pipeline on UK general election results (1918–2024).

## Contents

| File | What it is |
| --- | --- |
| `assignment-brief.pdf` | Official coursework brief |
| `task-1-constraint-solver.ipynb` | Task 1 — Futoshiki-style 5×5 grid solver |
| `task-2-election-prediction.ipynb` | Task 2 — cleaning, EDA, sklearn and PyTorch models on parliamentary data |

## What I did

**Task 1 (24%).** Implemented a constraint-propagation algorithm (uniqueness, row/column inequalities, pairs, triples, existence) to fill 5×5 grids under Latin-square and inequality constraints, and solved the four set configurations.

**Task 2 (76%).** Cleaned 18,000+ constituency-election rows, visualised vote structure, then predicted 2024 outcomes: logistic regression at ~75% accuracy, and a PyTorch classifier at ~80–85% on held-out 2024 seats. Linear regression for Labour vote share, plus equivalent cleaning in SQL.
