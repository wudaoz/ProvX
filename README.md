# ProvX (paper code)

This repository contains code and artifacts for **ProvX**, including model training/evaluation and an explanation pipeline over provenance data.

## Repository layout

```
.
├── main.py                    # Main entry point (train / test / explain)
├── dataset.py                 # Dataset utilities
├── graph_dataset.py           # Graph dataset helpers
├── line_extract.py            # Line/ground-truth extraction utilities
├── data_pre.py                # Data preprocessing helpers
│
├── models/                    # Models + explainers
│   ├── detector.py            # GNN-based detector model
│   ├── ProvX.py               # ProvX explainer (renamed from cfexplainer_1)
│   ├── gnnexplainer.py        # GNNExplainer wrapper
│   ├── pgexplainer.py         # PGExplainer wrapper
│   ├── subgraphx.py           # SubgraphX wrapper
│   ├── gnn_lrp.py             # LRP-based explainer
│   ├── deeplift.py            # DeepLIFT wrapper
│   ├── gradcam.py             # GradCAM wrapper
│   ├── shapley.py             # Shapley-style scoring utilities
│   └── nodoze*.py             # NoDoze baseline implementation
│
├── helpers/                   # Misc utilities (paths, joern helpers, etc.)
└── storage/                   # Cached data, processed artifacts, and external tools
```

## Key concepts / important files

- **`main.py`**
  - Implements the end-to-end workflow (training, testing, and explanation).
  - The explanation method is selected via `--ipt_method`.

- **`models/ProvX.py`**
  - The ProvX explainer implementation.
  - Exposes the same forward interface as other explainers used in `main.py`.


## Running (high level)

This codebase is research-oriented and expects datasets under `./Datasets/<dataset_name>/...` as referenced by `main.py`.

Common flags (see `main.py` for the full list):

- `--do_train`: train the detector model
- `--do_test`: evaluate on the test split
- `--do_explain`: run an explainer and evaluate explanations
- `--ipt_method provx`: run the ProvX explainer
- `--dataset_name <name>`: dataset identifier used for locating data and cache directories

Example (run explanation with ProvX):

```bash
python main.py --do_test --do_explain --ipt_method provx --dataset_name Cadets_com
```


