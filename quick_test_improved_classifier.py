import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from improved_classifier import NeuralRumorClassifier


def main():
    train_df = pd.read_csv("./data/train.csv")
    val_df = pd.read_csv("./data/val.csv")

    clf = NeuralRumorClassifier(threshold=0.51)

    print("Loading training data...")
    for _, row in train_df.iterrows():
        clf.update(str(row["text"]), str(row["label"]))

    print("Predicting validation data...")
    y_true = []
    y_pred = []

    for _, row in val_df.iterrows():
        pred = clf.predict(str(row["text"]))
        y_true.append(int(row["label"]))
        y_pred.append(int(pred.label))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print()
    print("===== Improved Classifier Test =====")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()