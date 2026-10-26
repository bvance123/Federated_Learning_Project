# Drift-Modulated Federated Learning for Heart Disease

ECE 4424 / CS 4824 course project (Fall 2026): Bennett Vance, Nikolay Rozenov, JJ Feeney.

We give each hospital a personal model trained with a drift penalty
`(mu/2) * ||v - w_global||^2`, and study how changing `mu` per hospital helps
small hospitals whose patients differ from the rest. Data: FLamby Fed-Heart-Disease (4 hospitals).

## Layout

| Path | What | Owner |
|---|---|---|
| `fl/model.py` | Shared network (`Model`), `get_weights` / `set_weights` | Bennett |
| `fl/server.py` | `aggregate` (FedAvg), `run_rounds` | Bennett |
| `fl/client.py` | `local_update`, personal models with mu | Nikolay |
| `fl/data.py` | `load_center`, preprocessing | JJ |
| `eval/` | Metrics (AUROC, sensitivity, calibration), figures | Bennett |
| `data/` | Downloaded FLamby data (not committed) | — |
| `results/` | Results CSVs | all |
| `LOG.md` | Problems and failed attempts, as they happen | all |

## Agreed interfaces

```python
load_center(center, seed) -> (train, val, test)    # each is (X [n,13] float, y [n] float 0/1)
Model()                                            # forward(x) -> P(disease), shape [n]
local_update(model, global_weights, train_data, mu, epochs) -> state_dict
aggregate(weights_list, sample_counts) -> state_dict
```

Results CSV columns: `method, center, seed, auroc, sensitivity, ece`

# Setup: FLamby and ECG Federated Learning

These instructions assume Linux or Fedora under WSL2, with Git and Conda/Miniconda installed.

## 1. Clone the project

If you have not cloned the project yet:

```bash
git clone https://github.com/bvance123/Federated_Learning_Project.git
cd Federated_Learning_Project
```

If you already have the repository, navigate to its root directory.

## 2. Create and activate the Conda environment

```bash
conda create -n ecg-fl python=3.10 pip -y
conda activate ecg-fl
```

Verify the environment:

```bash
python --version
which python
```

The Python executable should be inside the `ecg-fl` Conda environment.

## 3. Install project dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
```

## 4. Clone and install FLamby

Clone FLamby alongside the project repository:

```bash
cd ..
git clone https://github.com/owkin/FLamby.git
cd FLamby
```

Install FLamby and its heart-disease dataset dependencies:

```bash
python -m pip install -e ".[heart]"
```

Verify the installation:

```bash
python -c "from flamby.datasets.fed_heart_disease import HeartDiseaseRaw; print('FLamby installed successfully')"
```

## 5. Download and configure the heart-disease dataset

From the FLamby repository:

```bash
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
```

Download the dataset:

```bash
python download.py --output-folder ./heart_disease_dataset
```

Configure FLamby to use the downloaded data:

```bash
python update_config.py --new-path "$(realpath ./heart_disease_dataset)"
```

Verify the downloaded files:

```bash
ls -lh heart_disease_dataset/
```

The directory should contain these four files:

- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

Each team member must download the dataset and configure its local path on their own machine.

## 6. Run the tests

Return to the project root:

```bash
cd ../Federated_Learning_Project
```

Run the project tests:

```bash
python -m fl.data_test          # data test
python -m pytest tests          # server tests, no data needed
```

## 7. Run the federated-learning experiment

From the project root, run the end-to-end FedAvg experiment:

```bash
python -m fl.server 
```

This requires the data loader, client, and server implementations to be configured correctly.

## Troubleshooting

- **`ModuleNotFoundError: flamby`**: Activate `ecg-fl` and install FLamby using `python -m pip install -e ".[heart]"` from the FLamby repository.
- **Dataset configuration or path error**: Repeat Step 5 and verify that the configured directory contains all four data files.
- **`ModuleNotFoundError: fl`**: Run project commands from the project root.
- **Conda command not found**: Initialize Conda for your shell or open a terminal where Conda is available.
- **No tests collected**: Verify that test files are located under `fl/` and use names recognized by pytest, such as `test_data.py` or `data_test.py`.

## Reproducibility

- Use Python 3.10 and the same dependency versions across experiments.
- Use consistent random seeds and train/validation/test splits.
- Keep FLamby and the project in separate repositories.
- Do not commit downloaded dataset files or machine-specific absolute paths.
- Cite the original UCI Heart Disease dataset and FLamby in project documentation.
