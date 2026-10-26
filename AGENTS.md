# University projects

MSc Data Science coursework, plus a Jupyter workspace for the VTaC ventricular-tachycardia alarm dataset.

## Cursor Cloud specific instructions

Python 3.12 is on the image. If `python3 -m venv .venv` fails because ensurepip is missing, install `python3.12-venv` first. The virtualenv is `.venv` at the repo root (gitignored).

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` uses the PyTorch CPU index. It includes numpy, pandas, matplotlib, scikit-learn, scipy, torch, JupyterLab, and wfdb.

JupyterLab for VTaC development, from the repo root:

```bash
source .venv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Starter notebook: `vtac/notebooks/vtac_deep_learning.ipynb`. It downloads the public label CSVs. It does not download the 2.7 GB waveform ZIP. Download and unpack instructions are in `vtac/README.md`. Set `MPLBACKEND=Agg` only for headless coursework notebooks; leave it unset when executing the VTaC notebook if you want the label chart embedded.

Other coursework notebooks:

- `AM41DP-data-science-programming/task-1-constraint-solver.ipynb` — numpy and math only.
- `AM41PR-visa-compliance-ml/final-pipeline.ipynb` — reads `Combined.csv` in that folder. Includes RandomizedSearchCV (`n_iter=50`) and GridSearchCV.
- `SMLassignmentFinalAC.ipynb` — reads `classification.csv` at the repo root. The committed file is `classification.csv.csv`; `classification.csv` is a symlink to it.
- `AM41DP-data-science-programming/task-2-election-prediction.ipynb` — reads `Election_results.csv`, which is not in the repo. Current scikit-learn also rejects `LogisticRegression(multi_class="ovr")`.
