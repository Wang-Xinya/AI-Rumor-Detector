import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to tfidf_mlp_predictions.csv")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "f1", "recall"])
    args = parser.parse_args()

    df = pd.read_csv(args.pred)

    y_true = df["true_label"].astype(int)
    prob_1 = df["prob_1"].astype(float)

    rows = []

    for i in range(5, 96):
        threshold = i / 100
        y_pred = (prob_1 >= threshold).astype(int)

        metrics = evaluate(y_true, y_pred)
        metrics["threshold"] = threshold
        rows.append(metrics)

    result = pd.DataFrame(rows)
    result = result[
        ["threshold", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    ]

    output_dir = os.path.dirname(args.pred)
    output_path = os.path.join(output_dir, "threshold_search.csv")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    best = result.sort_values(by=args.metric, ascending=False).iloc[0]

    print(f"Threshold search saved to: {output_path}")
    print()
    print(f"Best threshold by {args.metric}:")
    print(best.to_string())


if __name__ == "__main__":
    main()