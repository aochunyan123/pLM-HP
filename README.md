# Peptide Classification (ESM2 Residue Embeddings + BiLSTM)

This repo provides a simple pipeline:
1) extract **per-residue ESM2 embeddings** from FASTA  
2) train a **BiLSTM** classifier on the variable-length embeddings

---

## Environment

- Python 3.9+
- PyTorch
- ESM (facebookresearch/esm)
- Biopython, NumPy, scikit-learn, pandas, tqdm

Install:
```bash
pip install -r requirements.txt


1) Extract ESM2 embeddings
Input: FASTA
Output: pkl file: dict[str, np.ndarray], each array has shape (L, D) (BOS/EOS removed)

python scripts/extract_embeddings.py \
  --fasta data/raw/train.fasta \
  --out data/processed/train_embeddings.pkl \
  --device cuda \
  --layer 12 \
  --max_tokens 2000 \
  --fp16

Do the same for test:

python scripts/01_extract_esm2_embeddings.py \
  --fasta data/raw/test.fasta \
  --out data/processed/test_embeddings.pkl \
  --device cuda \
  --layer 12 \
  --max_tokens 2000 \
  --fp16


2) Train the BiLSTM classifier
Run from the project root:

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


