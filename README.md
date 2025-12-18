## pLM-HP

Peptide hormones are key regulators of metabolism, growth, and homeostasis, and they are important targets in peptide drug discovery. We propose **pLM-HP**, a deep learning framework that combines the ESM2 protein language model with a BiLSTM. Across benchmark evaluations, **pLM-HP** achieves accurate and well-balanced performance, outperforming existing peptide hormone predictors.

---

## Environment

python==3.10
torch==2.9.1
networkx==3.4.2
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
scipy==1.15.3



## Running

```bash
1) Extract ESM2 embeddings
Input: FASTA
Output: pkl file: dict[str, np.ndarray], each array has shape (L, D) 


python scripts/extract_embeddings.py \
  --fasta data/raw/train.fasta \
  --out data/processed/train_embeddings.pkl \
  --device cuda \
  --layer 12 \
  --max_tokens 2000 \
  --fp16


python scripts/extract_embeddings.py \
  --fasta data/raw/test.fasta \
  --out data/processed/test_embeddings.pkl \
  --device cuda \
  --layer 12 \
  --max_tokens 2000 \
  --fp16


2) Train the model

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

