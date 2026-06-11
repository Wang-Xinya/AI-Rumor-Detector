"""
torch_text_classifiers.py - Optional PyTorch text classifiers for experiments.

This module is not required by the main rumor detector. It is used by
compare_deep_models.py when torch is installed, so the default lightweight
scikit-learn workflow remains usable on machines without PyTorch.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


def tokenize(text: str) -> list[str]:
    """Tokenize short social media text with simple, reproducible rules."""
    return [m.group() for m in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fa5]|[?？!！]", (text or "").lower())]


@dataclass
class TorchModelResult:
    labels: list[str]
    probabilities: list[dict[str, float]]


if TORCH_AVAILABLE:

    class TextCNN(nn.Module):
        def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 96):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(embed_dim, 64, kernel_size=kernel_size) for kernel_size in (3, 4, 5)]
            )
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(64 * len(self.convs), num_classes)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids).transpose(1, 2)
            pooled = []
            for conv in self.convs:
                activated = torch.relu(conv(embedded))
                pooled.append(torch.max(activated, dim=2).values)
            features = self.dropout(torch.cat(pooled, dim=1))
            return self.fc(features)


    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 96, hidden_dim: int = 96):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids)
            _, (hidden, _) = self.lstm(embedded)
            features = torch.cat([hidden[-2], hidden[-1]], dim=1)
            return self.fc(self.dropout(features))


class TorchTextClassifier:
    """A small wrapper that trains either TextCNN or BiLSTM for comparison."""

    def __init__(
        self,
        architecture: str,
        max_vocab_size: int = 20000,
        max_len: int = 80,
        epochs: int = 8,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        if architecture not in {"textcnn", "bilstm"}:
            raise ValueError("architecture must be 'textcnn' or 'bilstm'")
        self.architecture = architecture
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.classes_: list[str] = []
        self.model = None

    def _require_torch(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install torch to run TextCNN/BiLSTM experiments.")

    def _build_vocab(self, texts: list[str]) -> None:
        counts = collections.Counter()
        for text in texts:
            counts.update(tokenize(text))

        for token, _ in counts.most_common(self.max_vocab_size - len(self.vocab)):
            self.vocab[token] = len(self.vocab)

    def _encode_one(self, text: str) -> list[int]:
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokenize(text)]
        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids.extend([self.vocab["<pad>"]] * (self.max_len - len(ids)))
        return ids

    def _encode_many(self, texts: list[str]) -> np.ndarray:
        return np.array([self._encode_one(text) for text in texts], dtype=np.int64)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self._require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = sorted({str(label) for label in labels})
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        self._build_vocab(texts)

        x = torch.tensor(self._encode_many(texts), dtype=torch.long)
        y = torch.tensor([class_to_idx[str(label)] for label in labels], dtype=torch.long)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.architecture == "textcnn":
            self.model = TextCNN(len(self.vocab), len(self.classes_))
        else:
            self.model = BiLSTMClassifier(len(self.vocab), len(self.classes_))
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()

        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

    def predict_many(self, texts: list[str]) -> TorchModelResult:
        self._require_torch()
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        device = next(self.model.parameters()).device
        x = torch.tensor(self._encode_many(texts), dtype=torch.long)
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)

        labels: list[str] = []
        probabilities: list[dict[str, float]] = []
        self.model.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                logits = self.model(batch_x.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for row in probs:
                    prob_map = {label: float(prob) for label, prob in zip(self.classes_, row)}
                    labels.append(max(prob_map, key=prob_map.get))
                    probabilities.append(prob_map)

        return TorchModelResult(labels=labels, probabilities=probabilities)
