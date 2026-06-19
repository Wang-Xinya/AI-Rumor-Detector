from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from text_features import clean_text, extract_text_stats
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)
    return df


def build_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    ngram_range=(1, 3),
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "style_stats",
                Pipeline(
                    [
                        ("stats", FunctionTransformer(extract_text_stats, validate=False)),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
            ),
        ]
    )
    
def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("features", build_features()),
            ("classifier", classifier),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    true_int = [int(x) for x in y_true]
    pred_int = [int(x) for x in y_pred]

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


def get_prob_1(model: Pipeline, texts: list[str]) -> np.ndarray:
    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "predict_proba"):
        probs = model.predict_proba(texts)
        classes = list(classifier.classes_)
        idx_1 = classes.index("1")
        return probs[:, idx_1]

    if hasattr(classifier, "decision_function"):
        scores = model.decision_function(texts)
        scores = np.asarray(scores, dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    preds = model.predict(texts)
    return np.array([1.0 if p == "1" else 0.0 for p in preds])


def threshold_search(y_true, prob_1, metric: str = "accuracy") -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = None

    for i in range(5, 96):
        threshold = i / 100
        y_pred = ["1" if p >= threshold else "0" for p in prob_1]
        metrics = evaluate(y_true, y_pred)
        metrics["threshold"] = threshold

        if best_metrics is None or metrics[metric] > best_metrics[metric]:
            best_metrics = metrics
            best_threshold = threshold

    return best_threshold, best_metrics


def save_predictions(output_dir, model_name, val_df, prob_1, threshold):
    pred = ["1" if p >= threshold else "0" for p in prob_1]

    details = val_df[["text", "label"]].copy()
    details = details.rename(columns={"label": "true_label"})
    details["pred_label"] = pred
    details["prob_0"] = 1.0 - prob_1
    details["prob_1"] = prob_1
    details["is_correct"] = details["true_label"].astype(str) == details["pred_label"].astype(str)

    details.to_csv(
        os.path.join(output_dir, f"{model_name}_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    fn = details[(details["true_label"] == "1") & (details["pred_label"] == "0")]
    fp = details[(details["true_label"] == "0") & (details["pred_label"] == "1")]

    fn.to_csv(
        os.path.join(output_dir, f"{model_name}_fn.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    fp.to_csv(
        os.path.join(output_dir, f"{model_name}_fp.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def main():
    train_path = "./data/train.csv"
    val_path = "./data/val.csv"

    train_df = load_data(train_path)
    val_df = load_data(val_path)

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    x_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/improved_model_experiment_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    model_configs = {
        "logreg_c1": LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "logreg_c2": LogisticRegression(
            C=2.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "logreg_c4": LogisticRegression(
            C=4.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "complement_nb": ComplementNB(alpha=0.3),
        "sgd_log": SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "linear_svc": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
        ),
        "mlp_v2": MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=5e-5,
            learning_rate_init=8e-4,
            max_iter=120,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        ),
    }

    trained_models = {}
    prob_map = {}
    summary_rows = []

    for model_name, classifier in model_configs.items():
        print(f"\nRunning {model_name}...")

        model = build_pipeline(clone(classifier))
        model.fit(x_train, y_train)

        prob_1 = get_prob_1(model, x_val)
        best_threshold, metrics = threshold_search(y_val, prob_1, metric="accuracy")

        metrics["model"] = model_name
        metrics["best_threshold"] = best_threshold
        summary_rows.append(metrics)

        trained_models[model_name] = model
        prob_map[model_name] = prob_1

        save_predictions(output_dir, model_name, val_df, prob_1, best_threshold)

        print(
            f"{model_name}: "
            f"threshold={best_threshold:.2f}, "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"FN={metrics['fn']}, FP={metrics['fp']}"
        )

    # 简单概率集成：把几个表现稳定且有概率输出的模型做加权平均
    ensemble_candidates = ["logreg_c2", "logreg_c4", "sgd_log", "complement_nb", "mlp_v2"]
    available = [name for name in ensemble_candidates if name in prob_map]

    if available:
        print("\nRunning ensemble_weighted...")

        weights = {
            "logreg_c2": 0.25,
            "logreg_c4": 0.25,
            "sgd_log": 0.20,
            "complement_nb": 0.10,
            "mlp_v2": 0.20,
        }

        total_weight = sum(weights[name] for name in available)
        ensemble_prob_1 = sum(prob_map[name] * weights[name] for name in available) / total_weight

        best_threshold, metrics = threshold_search(y_val, ensemble_prob_1, metric="accuracy")
        metrics["model"] = "ensemble_weighted"
        metrics["best_threshold"] = best_threshold
        summary_rows.append(metrics)

        save_predictions(output_dir, "ensemble_weighted", val_df, ensemble_prob_1, best_threshold)

        print(
            f"ensemble_weighted: "
            f"threshold={best_threshold:.2f}, "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"FN={metrics['fn']}, FP={metrics['fp']}"
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary[
        [
            "model",
            "best_threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ].sort_values(by="accuracy", ascending=False)

    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n===== Summary =====")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()