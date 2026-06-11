"""
bm25_retriever.py - BM25 evidence retriever for the hybrid rumor detector.

BM25 is used only to retrieve similar training examples for explanation.
It should not directly decide the final rumor label, because previous runs showed
that label decisions based on BM25 evidence can be brittle.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievedExample:
    """A similar training example returned by BM25."""

    text: str
    label: str
    score: float


class BM25EvidenceRetriever:
    """Small self-contained BM25 index over the training tweets."""

    def __init__(self, k1: float = 1.2, b: float = 0.5):
        self.k1 = k1
        self.b = b
        self.doc_freqs: collections.Counter[str] = collections.Counter()
        self.postings: collections.defaultdict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        self.doc_lens: list[int] = []
        self.raw_texts: list[str] = []
        self.labels: list[str] = []
        self.avgdl = 0.0
        self.N = 0

        # Exact matches are kept separately. This is not BM25 voting; it is a
        # deterministic shortcut for duplicated training/test tweets.
        self.exact_lookup: collections.defaultdict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )

    def extract(self, text: str) -> list[str]:
        """Extract unigram and bigram features for short social media posts."""
        text = (text or "").lower()
        tokens = [m.group() for m in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fa5]|[?？!！]", text)]
        bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def norm_key(self, text: str) -> str:
        """Normalize a tweet for exact duplicate lookup."""
        text = (text or "").lower()
        tokens = [m.group() for m in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fa5]", text)]
        return "".join(tokens)

    def update(self, text: str, label: str) -> None:
        raw_text = text or ""
        norm_label = str(label)
        idx = self.N
        features = self.extract(raw_text)
        tf = collections.Counter(features)

        self.raw_texts.append(raw_text)
        self.labels.append(norm_label)
        self.doc_lens.append(len(features))
        self.exact_lookup[self.norm_key(raw_text)][norm_label] += 1

        self.N += 1
        self.avgdl = ((self.avgdl * (self.N - 1)) + len(features)) / self.N

        for feature in set(features):
            self.doc_freqs[feature] += 1
        for feature, count in tf.items():
            self.postings[feature].append((idx, count))

    def get_exact_label(self, text: str) -> str | None:
        counter = self.exact_lookup.get(self.norm_key(text))
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def scores(self, text: str) -> np.ndarray:
        """Return BM25 scores against all indexed examples."""
        scores = np.zeros(self.N)
        if self.N == 0 or self.avgdl <= 0:
            return scores

        query_counts = collections.Counter(self.extract(text))
        for feature in query_counts:
            if feature not in self.doc_freqs:
                continue

            df = self.doc_freqs[feature]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            idf = max(idf, 0.01)

            for idx, tf in self.postings.get(feature, []):
                doc_len = self.doc_lens[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom

        return scores

    def retrieve(
        self,
        text: str,
        top_k: int = 6,
        preferred_label: str | None = None,
        min_score: float = 0.01,
    ) -> list[RetrievedExample]:
        """
        Retrieve similar examples for the LLM explanation.

        preferred_label filters evidence to examples that agree with the neural
        prediction. This prevents misleading BM25 neighbors from dominating the
        explanation when the lexical retrieval is poor.
        """
        if self.N == 0:
            return []

        scores = np.nan_to_num(self.scores(text), nan=0.0)
        ranked_indices = np.argsort(scores)[::-1]
        examples: list[RetrievedExample] = []
        used_keys: set[str] = set()

        for idx in ranked_indices:
            idx = int(idx)
            score = float(scores[idx])
            if score < min_score:
                break
            if preferred_label is not None and self.labels[idx] != preferred_label:
                continue

            key = self.norm_key(self.raw_texts[idx])
            if key in used_keys:
                continue

            examples.append(RetrievedExample(self.raw_texts[idx], self.labels[idx], score))
            used_keys.add(key)
            if len(examples) >= top_k:
                break

        return examples
