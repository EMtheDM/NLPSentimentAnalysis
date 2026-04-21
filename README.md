# NLP Sentiment Analysis Project

## Overview
This project performs sentiment analysis on product reviews by comparing a traditional machine learning approach (Logistic Regression with TF-IDF) against a transformer-based model (DistilBERT).

The goal is to evaluate whether modern transformer models provide meaningful improvements over simpler baseline models.

---

## How to Run the Project

### 1. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# or
.venv\Scripts\activate      # Windows
```

### 2. Install Dependencies

```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

---

### 3. Run the Project

Run both models:

```bash
python train_sentiment.py --run_logreg --run_bert
```

Run with smaller dataset:

```bash
python train_sentiment.py --run_logreg --run_bert --train_limit 5000 --test_limit 2000
```

Run only baseline model:

```bash
python train_sentiment.py --run_logreg
```

Run only BERT model:

```bash
python train_sentiment.py --run_bert
```

---

## Dataset

This project uses the Amazon Polarity dataset from Hugging Face:

- Loaded using:
  load_dataset("amazon_polarity")
- Labels:
  - 0 = negative
  - 1 = positive

---

## Project Structure

NLPSentimentProject/

- train_sentiment.py        # Main script
- outputs/
  - model_comparison.csv    # Results
- checkpoints/              # Saved models
- README.md                 # Documentation

---

## Models Used

### Logistic Regression + TF-IDF
- Fast and interpretable baseline

### DistilBERT
- Transformer-based model
- Captures context and meaning

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

Results saved to:
outputs/model_comparison.csv

---

## Notes

- First run downloads dataset and model
- BERT training takes longer than baseline
- Works on Apple Silicon using PyTorch MPS

---

## Summary

- Traditional models are fast and effective
- Transformer models perform better
- Context matters in NLP
