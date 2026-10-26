# VTaC deep-learning workspace

Jupyter environment for developing models on **VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors** (version 1.0).

The waveforms are not stored in this repository. Download them from PhysioNet, then open `notebooks/vtac_deep_learning.ipynb`.

## Download the dataset

Project page (description, citation, file list):

https://physionet.org/content/vtac/1.0/

Direct ZIP (about 2.7 GB). The downloaded filename is `vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-icu-monitors-1.0.zip`:

https://physionet.org/content/vtac/get-zip/1.0/

DOI: https://doi.org/10.13026/8td2-g363

License: Creative Commons Attribution-ShareAlike 4.0. No PhysioNet login is required.

Unpack the archive so the WFDB files land in `vtac/data/`:

```bash
mkdir -p vtac/data
unzip -q vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-icu-monitors-1.0.zip -d vtac/data
```

You should then have `vtac/data/waveforms/`, plus `event_labels.csv` and `benchmark_data_split.csv`. Unpacked size is about 4 GB. `vtac/data/` is gitignored.

The label tables alone are small. The starter notebook downloads them from:

- https://physionet.org/files/vtac/1.0/event_labels.csv?download
- https://physionet.org/files/vtac/1.0/benchmark_data_split.csv?download

Official training code (Python 3.8, PyTorch 1.9) lives at https://github.com/ML-Health/VTaC. This workspace uses the current CPU PyTorch in the repo virtualenv so new notebooks run on Python 3.12. Each recording is 6 minutes at 250 Hz. The paper's realtime models use the 10 seconds immediately before the alarm, which is sample range `[72500:75000]` on a 90,000-sample record, with four channels (two ECG leads, PPG/PLETH, and ABP).

## Run Jupyter

From the repository root, with `.venv` already created (see `AGENTS.md`):

```bash
source .venv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open `vtac/notebooks/vtac_deep_learning.ipynb`.
