"""
compare_deep_models.py - Compare neural classifiers for rumor detection.

This script evaluates classification only. It does not call the LLM, so it is
much faster than run.py and is suitable for model selection experiments.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from neural_classifier import NeuralRumorClassifier


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ["text", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing required column: {col}")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)
    return df


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float | int]:
    true_int = [int(label) for label in y_true]
    pred_int = [int(label) for label in y_pred]
    tn, fp, fn, tp = confusion_matrix(true_int, pred_int, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(true_int, pred_int),
        "precision": precision_score(true_int, pred_int, zero_division=0),
        "recall": recall_score(true_int, pred_int, zero_division=0),
        "f1": f1_score(true_int, pred_int, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_tfidf_mlp(train_df: pd.DataFrame, val_df: pd.DataFrame, threshold: float | None = None):
    classifier = NeuralRumorClassifier()
    for _, row in train_df.iterrows():
        classifier.update(row["text"], row["label"])

    predictions = []
    prob_rows = []
    for text in val_df["text"].tolist():
        prediction = classifier.predict(text)
        probs = prediction.probabilities
        label = prediction.label
        if threshold is not None and "1" in probs:
            label = "1" if probs.get("1", 0.0) >= threshold else "0"
        predictions.append(label)
        prob_rows.append(probs)

    return predictions, prob_rows


def run_torch_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, epochs: int):
    from torch_text_classifiers import TORCH_AVAILABLE, TorchTextClassifier

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed; skipping torch models.")

    classifier = TorchTextClassifier(architecture=model_name, epochs=epochs)
    classifier.fit(train_df["text"].tolist(), train_df["label"].tolist())
    result = classifier.predict_many(val_df["text"].tolist())
    return result.labels, result.probabilities


def probability_columns(prob_rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prob_0": [row.get("0", 0.0) for row in prob_rows],
            "prob_1": [row.get("1", 0.0) for row in prob_rows],
        }
    )


def save_model_outputs(
    output_dir: str,
    model_name: str,
    val_df: pd.DataFrame,
    predictions: list[str],
    prob_rows: list[dict[str, float]],
) -> None:
    details = val_df[["text", "label"]].copy()
    details = details.rename(columns={"label": "true_label"})
    details["pred_label"] = predictions
    details = pd.concat([details, probability_columns(prob_rows)], axis=1)
    details["is_correct"] = details["true_label"].astype(str) == details["pred_label"].astype(str)

    details_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    details.to_csv(details_path, index=False, encoding="utf-8")

    false_negative = details[(details["true_label"].astype(str) == "1") & (details["pred_label"].astype(str) == "0")]
    false_positive = details[(details["true_label"].astype(str) == "0") & (details["pred_label"].astype(str) == "1")]
    false_negative.to_csv(os.path.join(output_dir, f"{model_name}_fn.csv"), index=False, encoding="utf-8")
    false_positive.to_csv(os.path.join(output_dir, f"{model_name}_fp.csv"), index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare neural rumor classifiers without LLM calls.")
    parser.add_argument("--train", default="./data/train.csv", help="Path to train CSV.")
    parser.add_argument("--val", default="./data/val.csv", help="Path to validation CSV.")
    parser.add_argument("--output-dir", default=None, help="Directory for comparison results.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional quick-test train subset size.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional quick-test validation subset size.")
    parser.add_argument("--torch-epochs", type=int, default=8, help="Epochs for TextCNN/BiLSTM if torch is installed.")
    parser.add_argument(
        "--models",
        default="tfidf_mlp,tfidf_mlp_t045,textcnn,bilstm",
        help="Comma-separated models: tfidf_mlp, tfidf_mlp_t045, textcnn, bilstm.",
    )
    args = parser.parse_args()

    train_df = load_data(args.train)
    val_df = load_data(args.val)
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=42)
    if args.max_val_samples:
        val_df = val_df.sample(n=min(args.max_val_samples, len(val_df)), random_state=42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"./results/deep_model_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    requested_models = [model.strip() for model in args.models.split(",") if model.strip()]
    summary_rows = []

    for model_name in requested_models:
        print(f"\nRunning {model_name}...")
        try:
            if model_name == "tfidf_mlp":
                predictions, prob_rows = run_tfidf_mlp(train_df, val_df)
            elif model_name == "tfidf_mlp_t045":
                # Lowering the rumor threshold is a targeted diagnostic for the
                # observed false-negative bias on rumor examples.
                predictions, prob_rows = run_tfidf_mlp(train_df, val_df, threshold=0.45)
            elif model_name in {"textcnn", "bilstm"}:
                predictions, prob_rows = run_torch_model(model_name, train_df, val_df, args.torch_epochs)
            else:
                print(f"Unknown model '{model_name}', skipped.")
                continue
        except RuntimeError as exc:
            print(f"{model_name} skipped: {exc}")
            continue

        metrics = evaluate_predictions(val_df["label"].tolist(), predictions)
        metrics["model"] = model_name
        summary_rows.append(metrics)
        save_model_outputs(output_dir, model_name, val_df, predictions, prob_rows)

        print(
            f"{model_name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"FN={metrics['fn']}, FP={metrics['fp']}"
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary[
            ["model", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
        ].sort_values(by="f1", ascending=False)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
