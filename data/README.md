# Dataset

This project uses the **Jigsaw Toxic Comment Classification** dataset from Kaggle.

## Download

1. Go to: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
2. Download `train.csv` and place it in this `data/` folder.

## Format

The CSV must have these columns:

| Column | Description |
|---|---|
| `id` | Unique comment ID |
| `comment_text` | Raw comment text |
| `toxic` | 0 or 1 |
| `severe_toxic` | 0 or 1 |
| `obscene` | 0 or 1 |
| `threat` | 0 or 1 |
| `insult` | 0 or 1 |
| `identity_hate` | 0 or 1 |

## Fine-tuning

After placing `train.csv` here:

```bash
python scripts/train.py --data data/train.csv --output models/bert-toxic-finetuned
```
