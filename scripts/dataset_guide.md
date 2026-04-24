# Dataset Guide for Toxic Comment Detection

## ✅ Current Status: Sample Dataset Ready

A sample dataset with 43 comments has been created at `data/train.csv` with:
- **22 toxic comments** (51.2%)
- **21 non-toxic comments** (48.8%)
- **6 toxicity categories**: toxic, severe_toxic, obscene, threat, insult, identity_hate

## 📥 Getting the Full Jigsaw Dataset

### Option 1: Kaggle Download (Recommended)

1. **Create Kaggle Account**: https://www.kaggle.com/
2. **Go to Competition**: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
3. **Download Files**:
   - `train.csv` (159,571 comments)
   - `test.csv` (optional, for testing)
   - `test_labels.csv` (optional)

4. **Place Files**: Move `train.csv` to the `data/` folder

### Option 2: Using Kaggle API

```bash
# Install Kaggle API
pip install kaggle

# Download dataset (requires API key setup)
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

# Extract and move train.csv
unzip jigsaw-toxic-comment-classification-challenge.zip
cp train.csv data/
```

### Option 3: Alternative Sources

If Kaggle is not accessible, try these alternatives:

1. **GitHub Mirror**: Search for "jigsaw toxic comment dataset" on GitHub
2. **Google Dataset Search**: https://datasetsearch.research.google.com/
3. **HuggingFace Datasets**: 
   ```python
   from datasets import load_dataset
   dataset = load_dataset("jigsaw_toxicity_classification")
   ```

## 🎯 Dataset Format Requirements

Your `train.csv` must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Unique comment ID |
| `comment_text` | String | The actual comment text |
| `toxic` | 0/1 | General toxicity |
| `severe_toxic` | 0/1 | Severe toxicity |
| `obscene` | 0/1 | Obscene content |
| `threat` | 0/1 | Threatening language |
| `insult` | 0/1 | Insulting content |
| `identity_hate` | 0/1 | Identity-based hate |

## 🚀 Training the Model

### With Sample Dataset (Quick Test)

```bash
# Test training with sample data
python scripts/train.py --data data/train.csv --output models/sample-trained --epochs 5
```

### With Full Dataset (Production)

```bash
# Full training with Jigsaw dataset
python scripts/train.py \
    --data data/train.csv \
    --output models/bert-toxic-finetuned \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 16 | Training batch size |
| `--lr` | 2e-5 | Learning rate |
| `--model_name` | bert-base-uncased | Base model to fine-tune |

## 📊 Expected Training Results

### Sample Dataset (43 comments)
- **Fast training** (~2-5 minutes)
- **Limited accuracy** (small dataset)
- **Good for testing** the pipeline

### Full Jigsaw Dataset (159,571 comments)
- **Longer training** (~30-60 minutes on CPU, ~10-20 minutes on GPU)
- **High accuracy** (ROC-AUC > 0.95)
- **Production-ready** model

## 🔧 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `--batch_size` to 8 or 4
2. **Slow Training**: Use GPU or reduce `--epochs`
3. **Poor Accuracy**: Increase `--epochs` or use full dataset

### GPU Acceleration

If you have NVIDIA GPU:
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Training will automatically use GPU
python scripts/train.py --data data/train.csv --output models/bert-toxic-finetuned
```

## 📈 Model Evaluation

After training, test your model:

```python
# Test the trained model
python -c "
from backend.model import ToxicityClassifier
classifier = ToxicityClassifier('models/bert-toxic-finetuned')
result = classifier.predict('You are stupid!')
print(result)
"
```

## 🎉 Next Steps

1. **Start with sample dataset** to verify everything works
2. **Download full dataset** for production model
3. **Train and evaluate** your fine-tuned model
4. **Update model path** in `backend/model.py` to use your trained model

---

*Summer of AI · Swecha × IIIT Hyderabad*
