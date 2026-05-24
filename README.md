# FreqDiff


## Directory Structure

```text
FreqDiff_clean_*/
├── src/
│   ├── main.py          # training and evaluation entry point
│   ├── model.py         # FreqDiff model
│   ├── diffusion.py     # diffusion decoder
│   ├── utils_entity.py  # data utilities and ranking metrics
│   ├── knowledge_graph.py
│   ├── regcn.py
│   └── step_sample.py
├── data/
│   ├── ICEWS14/
│   ├── ICEWS18/
│   ├── ICEWS05-15/
│   └── GDELT/
├── checkpoints/         # generated during training; empty in this package
├── logs/                # generated during training/evaluation; empty in this package
└── results/             # generated prediction/ranking logs; empty in this package
```
Dataset can be obtained from [DIffuTKG]([https://arxiv.org/abs/2312.12021](https://github.com/AONE-NLP/DiffuTKG))

Each dataset folder contains the raw temporal quadruple splits (`train.txt`, `valid.txt`, `test.txt`), dictionaries (`entity2id.txt`, `relation2id.txt`, `stat.txt`), and precomputed `history_seq/` files used by the model.

## Environment

A CUDA-enabled PyTorch environment is recommended.

Required Python packages include:

```text
python >= 3.8
torch
numpy
scipy
pandas
tqdm
fitlog
dgl
torch-scatter
```

Install the PyTorch, DGL, and torch-scatter builds that match your CUDA version.

## Basic Usage

Run commands from the package root.

Train on ICEWS14:

```bash
python src/main.py --dataset ICEWS14 --gpu 0 --seed 2026
```

Evaluate a saved ICEWS14 checkpoint:

```bash
python src/main.py --dataset ICEWS14 --gpu 0 --test
```


Model checkpoints are written to `checkpoints/<dataset>_fre_<fre>/`. Logs are written to `logs/data_<dataset>/`, and prediction outputs are written to `results/`.
