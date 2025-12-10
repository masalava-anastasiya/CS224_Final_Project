## CS224W Final Project — Predicting Disease Trajectories with Temporal Graph Networks

Temporal link prediction on a disease progression graph built from MIMIC‑IV diagnosis histories. We compare temporal graph neural networks (TGN, TGAT‑style) against simple but strong baselines to predict the next disease given a patient’s history.

### Motivation

Health conditions often occur in sequences rather than isolation. Modeling disease trajectories as a graph lets us learn from comorbidities and temporal patterns instead of treating diagnoses independently. Graph Neural Networks (GNNs) propagate information over disease–disease edges, and temporal GNNs add timing and order, which are critical in clinical data.

---

## Repository Structure
- `cs224w_baselines.ipynb` — Random, Most Frequent Next Hop, Degree‑Aware baselines.
- `TGN_model.ipynb` — Temporal Graph Network (TGN) training/evaluation.
- `TGAT_model.ipynb` — TGAT‑style model (Transformer attention over temporal neighbors) training/evaluation.

---

## Data

- **Source**: MIMIC‑IV EHR (Beth Israel Deaconess Medical Center, 2008–2019). Access requires credentialing via PhysioNet.
- **Preprocessed data (download)**: Please download `disease_nodes.csv` and `disease_edges.csv` from the Google Drive folder and place them locally (e.g., `mimic_preprocessed/`) before running the notebooks:
  - Google Drive: [Disease graph CSVs](https://drive.google.com/drive/folders/1lEbimz8E2Hk2ii0a7y6qqnS-liNDJ8YI?usp=share_link)
  - Nodes: unique ICD codes.
  - Edges: directed disease transitions with timestamps, constructed by ordering patient diagnoses chronologically and adding `A → B` when `B` occurs after `A`.
---

## Setup

You can use any recent Python (3.9–3.11). A clean environment is recommended.

### 1) Create environment

```bash
conda create -n cs224w-gnn python=3.10 -y
conda activate cs224w-gnn
```

### 2) Install core dependencies

Install PyTorch (CPU example below; for CUDA choose the commands from the official site).

```bash
# PyTorch (CPU example). For CUDA, follow the official selector.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Common scientific stack
pip install numpy pandas scikit-learn matplotlib tqdm jupyter networkx
```

Install PyTorch Geometric (PyG). Because wheels depend on your exact torch/CUDA version, prefer the official instructions:
- PyTorch: [official install selector](https://pytorch.org/get-started/locally/)
- PyG: [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

For CPU‑only with recent torch, a typical pattern is:

```bash
# Example for PyTorch 2.x CPU-only (adjust URL for your torch version)
pip install torch-geometric
```

If you encounter build/wheel errors on `torch-scatter`, `torch-sparse`, etc., use the wheel index suggested by the PyG install guide that matches your Torch/CUDA.

### Running at Scale (Recommended: Vertex AI)

Because the dataset is large and training can be compute‑intensive, we recommend running the notebooks on **Google Cloud Vertex AI Workbench**:
- Spin up a Managed Notebook with sufficient CPU/GPU and memory (e.g., T4/A100 if available).
- Clone this repository into the notebook environment.
- Reuse the Setup steps above to install Python packages inside the notebook.
- Optionally attach a Cloud Storage bucket for datasets/checkpoints if you rerun preprocessing or log artifacts.

---

## How to Run

1) Start Jupyter:
```bash
jupyter lab
# or
jupyter notebook
```

2) Open the notebooks in this order (paths relative to repo root):
- `cs224w_baselines.ipynb` — runs:
  - **Completely Random**: uniform scores and random top‑K.
  - **Most Frequent Next Hop**: ranks `P(dst | src)` by empirical transition frequency from training edges.
  - **Degree‑Aware**: ranks by global in‑degree popularity of `dst`.
- `TGN_model.ipynb` — trains and evaluates a **Temporal Graph Network** with per‑node memory and time encodings.
- `TGAT_model.ipynb` — trains and evaluates a **TGAT‑style** attention model over recent temporal neighbors.

3) Data paths:
- Notebooks expect `mimic_preprocessed/disease_nodes.csv` and `mimic_preprocessed/disease_edges.csv` by default. Update the paths in the first cells if you have a different layout.

---

## Task and Metrics

- **Task**: temporal link prediction — given source disease and time, rank candidate next diseases.
- **Metrics**:
  - **AUC‑ROC** — global separability.
  - **Average Precision (AP)** — emphasizes high‑rank correctness.
  - **Hits@10** — does the true next disease appear in top‑10? A practical clinical‑ranking metric.

---

## Baselines

- **Completely Random**: structure‑agnostic lower bound.
- **Most Frequent Next Hop**: `P(dst | src)` from training transitions.
- **Degree‑Aware**: global popularity via `in_degree(dst)`, normalized.

These isolate where performance comes from: chance, local transition statistics, or global hub bias.

---

## Models

### Temporal Graph Network (TGN)
Maintains a learned **memory state per node** that is updated as timestamped edges arrive (message creation → aggregation → memory update → embedding with time encoding). This captures long‑term context and irregular time gaps common in clinical trajectories.

### TGAT‑style (Temporal Graph Attention)
Adds multi‑head **attention over recent temporal neighbors** using time encodings so each event can emphasize relevant historical interactions. We use a two‑layer design to reach neighbors‑of‑neighbors with residual connections, and combine this with the TGN memory for a hybrid long‑/short‑term view.

---

## Acknowledgements

This work was conducted as part of Stanford CS224W (Machine Learning with Graphs). We thank the MIMIC‑IV team and course staff. Image credits as cited in the blog/lecture materials.

