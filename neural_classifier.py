"""
neural_classifier.py - Lightweight neural rumor classifier.

The classifier uses TF-IDF features followed by a multi-layer perceptron (MLP).
It is intentionally local and reproducible: no external model download is needed,
while the hidden layers still provide a real neural component for the hybrid system.
"""

from __future__ import annotations

import collections
import warnings
from dataclasses import dataclass

from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline


@dataclass
class NeuralPrediction:
    """Prediction returned by the neural classifier."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class NeuralRumorClassifier:
    """
    TF-IDF + MLP rumor classifier.

    Training is lazy: samples are collected through update(), and the neural
    network is fitted on the first predict(). This keeps compatibility with the
    original Harness API, where run.py calls update() repeatedly before predict().
    """

    def __init__(self, random_state: int = 42):
        self.texts: list[str] = []
        self.labels: list[str] = []
        self.label_counts: collections.Counter[str] = collections.Counter()
        self._is_dirty = False
        self._pipeline = self._build_pipeline(random_state)

    def _build_pipeline(self, random_state: int) -> Pipeline:
        features = FeatureUnion(
            [
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        max_features=15000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=12000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=random_state,
        )

        return Pipeline([("features", features), ("classifier", classifier)])

    def update(self, text: str, label: str) -> None:
        self.texts.append(text or "")
        self.labels.append(str(label))
        self.label_counts[str(label)] += 1
        self._is_dirty = True

    @property
    def is_ready(self) -> bool:
        return bool(self.texts) and len(self.label_counts) >= 2

    def fit_if_needed(self) -> None:
        if not self._is_dirty or not self.is_ready:
            return
        min_class_count = min(self.label_counts.values())
        enough_validation_data = len(self.labels) >= 20 and min_class_count >= 2
        # MLP early_stopping internally creates a stratified validation split.
        # Very small toy datasets cannot satisfy that split, so disable it there.
        self._pipeline.set_params(classifier__early_stopping=enough_validation_data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._pipeline.fit(self.texts, self.labels)
        self._is_dirty = False

    def predict(self, text: str) -> NeuralPrediction:
        """
        Predict a label and confidence.

        If the training data unexpectedly contains only one class, fall back to
        the majority class instead of failing during MLP training.
        """
        if not self.label_counts:
            return NeuralPrediction(label="", confidence=0.0, probabilities={})

        if not self.is_ready:
            label, count = self.label_counts.most_common(1)[0]
            confidence = count / max(sum(self.label_counts.values()), 1)
            return NeuralPrediction(label=label, confidence=confidence, probabilities={label: confidence})

        self.fit_if_needed()

        try:
            probabilities = self._pipeline.predict_proba([text or ""])[0]
            classes = self._pipeline.named_steps["classifier"].classes_
        except NotFittedError:
            label, count = self.label_counts.most_common(1)[0]
            confidence = count / max(sum(self.label_counts.values()), 1)
            return NeuralPrediction(label=label, confidence=confidence, probabilities={label: confidence})

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }
        label = max(probability_map, key=probability_map.get)
        return NeuralPrediction(
            label=label,
            confidence=probability_map[label],
            probabilities=probability_map,
        )
