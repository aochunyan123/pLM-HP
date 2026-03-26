## pLM-HP

Peptide hormones are key regulators of metabolism, growth, and homeostasis, and represent important targets in peptide drug discovery. Here, we propose **pLM-HP**, a deep learning framework that combines the ESM2 protein language model with a BiLSTM. Across benchmark evaluations, **pLM-HP** achieves accurate and well-balanced performance and outperforms existing peptide hormone predictors.

---


## Requirements

The framework was developed and tested with the following environment:

- Python 3.10
- torch 2.9.1
- networkx 3.4.2
- numpy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- scipy 1.15.3
- fair-esm 2.0.0
- biopython 1.86

**Note:** The required dependencies are listed in `requirements.txt` for standard one-command installation.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/aochunyan123/pLM-HP.git
cd pLM-HP
pip install -r requirements.txt
```

## Usage
### 1. Extract ESM2 embeddings

**Input:** FASTA files  
**Output:** PKL files in the format `dict[str, np.ndarray]`, where each array has shape `(L, D)`

Extract embeddings for the training set:
```bash
python scripts/extract_embeddings.py \
  --fasta data/raw/train.fasta \
  --out data/processed/train_embeddings.pkl \
  --device cuda \
  --layer 12 \
  --max_tokens 2000 \
  --fp16
```

Extract embeddings for the test set:
```bash
python scripts/extract_embeddings.py \
  --fasta data/raw/test.fasta \
  --out data/processed/test_embeddings.pkl \
  --device cuda \
  --layer 12 \
  --max_tokens 2000 \
  --fp16
```

### 2. Model training

```bash
python -m scripts.train \
  --train_pkl data/processed/train_embeddings.pkl \
  --test_pkl  data/processed/test_embeddings.pkl \
  --device cuda \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --use_minmax \
  --save_metric BACC
```
