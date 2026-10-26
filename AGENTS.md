# University projects

MSc Data Science coursework. The runnable pieces are Jupyter notebooks. Probabilistic modelling and the CS4730 brief are PDF reports only.

## Cursor Cloud specific instructions

Python 3.12 is already on the image. `python3-venv` is not, so install it before creating the virtualenv. Nothing in this repo starts a server.

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins numpy, pandas, matplotlib, scikit-learn, scipy, a CPU PyTorch wheel, jupyter, nbconvert, and ipykernel. The install is idempotent. Set `MPLBACKEND=Agg` before executing notebooks from a terminal.

Notebooks, from the directory that contains each file:

- `AM41DP-data-science-programming/task-1-constraint-solver.ipynb` — numpy and math only. Runs as checked in. Three of the four grids solve; the second configuration stops with "No progress made in this iteration", which is the notebook's own result.
- `AM41PR-visa-compliance-ml/final-pipeline.ipynb` — reads `Combined.csv` in that folder (3,957 rows). Full RandomizedSearchCV (`n_iter=50`) then GridSearchCV is part of the notebook. Do not shrink the search.
- `SMLassignmentFinalAC.ipynb` — reads `classification.csv` from the repo root. The committed data file is `classification.csv.csv`. A symlink `classification.csv` -> `classification.csv.csv` is in the repo. If the link is missing: `ln -s classification.csv.csv classification.csv`. The import cell originally sat below the first `pd.read_csv`, so a fresh kernel raised `NameError: pd`. That import cell now comes first.
- `AM41DP-data-science-programming/task-2-election-prediction.ipynb` — `pd.read_csv("Election_results.csv")` relative to that folder. The CSV is not in the repo. The saved notebook output matches the House of Commons Library constituency file (CBP-8647: `constituency_name`, party vote shares, `election`, `boundary_set`), but the public file ends in 2019, includes a `seats` column, and has trailing spaces on some headers. This notebook trains on years before 2024 and tests on 2024, and it calls `LogisticRegression(multi_class="ovr")`, which scikit-learn 1.9 no longer accepts. No course CSV was added. Imports of numpy, pandas, matplotlib, sklearn, and torch still succeed.

Example:

```bash
export MPLBACKEND=Agg
.venv/bin/jupyter nbconvert --execute --to notebook \
  --ExecutePreprocessor.timeout=1800 \
  AM41DP-data-science-programming/task-1-constraint-solver.ipynb
```

VTaC / PhysioNet is not part of this repository. Do not clone or download that dataset for this codebase.
