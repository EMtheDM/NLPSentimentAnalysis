import argparse
import os
import re
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


# -----------------------------
# Utilities
# -----------------------------
def clean_text(text: str) -> str:
    """Basic cleaning for traditional ML baseline."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics_from_preds(y_true, y_pred) -> dict:
    """Return standard evaluation metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"{k.capitalize():>10}: {v:.4f}")


def save_confusion_matrix(y_true, y_pred, out_path: str, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Data loading
# -----------------------------
@dataclass
class DataBundle:
    train_texts: list
    test_texts: list
    train_labels: list
    test_labels: list


def load_local_csv(
    path: str,
    text_col: str,
    label_col: Optional[str] = None,
    stars_col: Optional[str] = None,
    drop_neutral: bool = True,
    test_size: float = 0.2,
    seed: int = 42,
) -> DataBundle:
    """
    Load local CSV.
    Supports either:
      - binary label column already in {0,1}
      - star column, converted to binary:
          1-2 => 0
          4-5 => 1
          3 => dropped if drop_neutral=True
    """
    df = pd.read_csv(path)

    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' in CSV.")

    df = df[[text_col] + ([label_col] if label_col else []) + ([stars_col] if stars_col else [])].copy()
    df = df.dropna(subset=[text_col])

    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"Missing label column '{label_col}' in CSV.")
        df["label"] = df[label_col].astype(int)
    elif stars_col:
        if stars_col not in df.columns:
            raise ValueError(f"Missing stars column '{stars_col}' in CSV.")
        df[stars_col] = pd.to_numeric(df[stars_col], errors="coerce")
        df = df.dropna(subset=[stars_col])

        if drop_neutral:
            df = df[df[stars_col] != 3]

        df["label"] = df[stars_col].apply(lambda x: 1 if x >= 4 else 0)
    else:
        raise ValueError("Provide either label_col or stars_col.")

    df["text"] = df[text_col].astype(str)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]

    train_df, test_df = train_test_split(
        df[["text", "label"]],
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    return DataBundle(
        train_texts=train_df["text"].tolist(),
        test_texts=test_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        test_labels=test_df["label"].tolist(),
    )


def load_amazon_hf(
    train_limit: Optional[int] = 5000,
    test_limit: Optional[int] = 2000,
    seed: int = 42,
) -> DataBundle:

    ds = load_dataset("amazon_polarity")

    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    if train_limit:
        train_df = train_df.sample(n=min(train_limit, len(train_df)), random_state=seed)
    if test_limit:
        test_df = test_df.sample(n=min(test_limit, len(test_df)), random_state=seed)

    return DataBundle(
        train_texts=train_df["content"].tolist(),  # IMPORTANT: column name is different
        test_texts=test_df["content"].tolist(),
        train_labels=train_df["label"].tolist(),
        test_labels=test_df["label"].tolist(),
    )


# -----------------------------
# Logistic Regression baseline
# -----------------------------
def run_logreg_baseline(data: DataBundle, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    train_clean = [clean_text(x) for x in data.train_texts]
    test_clean = [clean_text(x) for x in data.test_texts]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_clean)
    X_test = vectorizer.transform(test_clean)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, data.train_labels)
    preds = model.predict(X_test)

    metrics = compute_metrics_from_preds(data.test_labels, preds)
    print_metrics("Logistic Regression + TF-IDF Results", metrics)
    print("\nClassification Report:\n")
    print(classification_report(data.test_labels, preds, target_names=["negative", "positive"], zero_division=0))

    save_confusion_matrix(
        data.test_labels,
        preds,
        os.path.join(output_dir, "logreg_confusion_matrix.png"),
        "LogReg Confusion Matrix",
    )

    return metrics


# -----------------------------
# BERT
# -----------------------------
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def run_bert_model(
    data: DataBundle,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = Dataset.from_dict({"text": data.train_texts, "label": data.train_labels})
    test_ds = Dataset.from_dict({"text": data.test_texts, "label": data.test_labels})
    dataset = DatasetDict({"train": train_ds, "test": test_ds})

    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics_from_preds(labels, preds)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "bert_checkpoints"),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_metrics,
    )

    trainer.train()
    predictions = trainer.predict(tokenized["test"])
    preds = np.argmax(predictions.predictions, axis=-1)

    metrics = compute_metrics_from_preds(data.test_labels, preds)
    print_metrics(f"{model_name} Results", metrics)
    print("\nClassification Report:\n")
    print(classification_report(data.test_labels, preds, target_names=["negative", "positive"], zero_division=0))

    save_confusion_matrix(
        data.test_labels,
        preds,
        os.path.join(output_dir, "bert_confusion_matrix.png"),
        "BERT Confusion Matrix",
    )

    return metrics


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Project: TF-IDF+LogReg vs BERT")

    parser.add_argument("--dataset", choices=["imdb", "csv"], default="imdb")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--stars_col", type=str, default=None)
    parser.add_argument("--drop_neutral", action="store_true")

    parser.add_argument("--train_limit", type=int, default=5000)
    parser.add_argument("--test_limit", type=int, default=2000)

    parser.add_argument("--run_logreg", action="store_true")
    parser.add_argument("--run_bert", action="store_true")

    parser.add_argument("--bert_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.run_logreg and not args.run_bert:
        args.run_logreg = True
        args.run_bert = True

    if args.dataset == "imdb":
        data = load_amazon_hf(
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            seed=args.seed,
        )
    else:
        if not args.csv_path:
            raise ValueError("For --dataset csv, provide --csv_path.")
        data = load_local_csv(
            path=args.csv_path,
            text_col=args.text_col,
            label_col=args.label_col,
            stars_col=args.stars_col,
            drop_neutral=args.drop_neutral,
            seed=args.seed,
        )

    print(f"\nTrain size: {len(data.train_texts)}")
    print(f"Test size:  {len(data.test_texts)}")
    print(f"Train positive ratio: {sum(data.train_labels)/len(data.train_labels):.3f}")
    print(f"Test positive ratio:  {sum(data.test_labels)/len(data.test_labels):.3f}")

    summary_rows = []

    if args.run_logreg:
        logreg_metrics = run_logreg_baseline(data, args.output_dir)
        summary_rows.append({"model": "LogReg+TFIDF", **logreg_metrics})

    if args.run_bert:
        bert_metrics = run_bert_model(
            data,
            args.output_dir,
            model_name=args.bert_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        summary_rows.append({"model": args.bert_model, **bert_metrics})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved comparison results to:", summary_path)
    print("\nFinal Summary:\n")
    print(summary_df)


if __name__ == "__main__":
    main()