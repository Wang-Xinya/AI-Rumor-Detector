"""
detection.py - Hybrid explainable rumor detection harness.

Final label: local neural classifier.
Evidence: BM25 retrieves similar training examples that agree with the neural label.
Explanation: school-provided LLM explains the fixed neural decision.
"""

from __future__ import annotations

import re

from bm25_retriever import BM25EvidenceRetriever, RetrievedExample
from harness_base import Harness
from improved_classifier import NeuralPrediction, NeuralRumorClassifier


class RumorDetectionHarness(Harness):
    """
    Explainable rumor detection Harness.

    This class deliberately separates responsibilities:
    - NeuralRumorClassifier produces the final 0/1 label.
    - BM25EvidenceRetriever supplies similar examples only as explanation evidence.
    - The LLM writes the natural-language rationale and should not change the label.
    """

    def __init__(self, call_llm, evidence_top_k: int = 5, use_exact_match: bool = True):
        super().__init__(call_llm)
        self.classifier = NeuralRumorClassifier()
        self.retriever = BM25EvidenceRetriever()
        self.evidence_top_k = evidence_top_k
        self.use_exact_match = use_exact_match

        # Kept for compatibility with the original run.py progress output.
        self.N = 0
        self.all_labels: set[str] = set()
        self.last_explanation = ""

    def update(self, text: str, label: str) -> None:
        """Store one labeled training sample and update both model components."""
        super().update(text, label)
        norm_label = str(label)
        self.classifier.update(text, norm_label)
        self.retriever.update(text, norm_label)
        self.N += 1
        self.all_labels.add(norm_label)

    def _parse_reasoning(self, response: str) -> str:
        """Extract <reasoning>...</reasoning>; fall back to raw text if needed."""
        if not response:
            return ""
        match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _format_evidence(self, examples: list[RetrievedExample]) -> str:
        if not examples:
            return "No reliable same-label BM25 evidence was found."

        lines = []
        for i, example in enumerate(examples, start=1):
            # Keep evidence compact so the prompt stays below the API context limit.
            text = example.text.replace("\n", " ").strip()
            if len(text) > 240:
                text = text[:237] + "..."
            lines.append(
                f"{i}. Post: {text}\n"
                f"   Label: {example.label}\n"
                f"   BM25 score: {example.score:.3f}"
            )
        return "\n".join(lines)

    def _make_explanation_messages(
        self,
        text: str,
        prediction: NeuralPrediction,
        evidence: list[RetrievedExample],
        decision_source: str,
    ) -> list[dict[str, str]]:
        """
        Build an explanation-only prompt.

        The label is fixed before calling the LLM. This avoids the old weakness
        where poor BM25 neighbors could make the LLM flip the classification.
        """
        label_name = "rumor" if prediction.label == "1" else "non-rumor"
        system_content = (
            "You are an expert assistant for explainable rumor detection. "
            "The final label has already been produced by a local neural classifier. "
            "Your job is to explain that fixed decision in a concise, faithful way. "
            "Do not change the label. Output only <reasoning>...</reasoning>."
        )

        user_content = (
            f"Post:\n{text}\n\n"
            f"Fixed final label: {prediction.label} ({label_name})\n"
            f"Decision source: {decision_source}\n"
            f"Neural confidence: {prediction.confidence:.3f}\n"
            f"Class probabilities: {prediction.probabilities}\n\n"
            "Same-label BM25 evidence from training data:\n"
            f"{self._format_evidence(evidence)}\n\n"
            "Write 2-4 sentences explaining why this post is classified this way. "
            "Use the evidence only as supporting context; if the evidence is weak, "
            "explain mainly from the post content and confidence."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _fallback_explanation(
        self,
        text: str,
        prediction: NeuralPrediction,
        evidence: list[RetrievedExample],
        decision_source: str,
    ) -> str:
        label_name = "rumor" if prediction.label == "1" else "non-rumor"
        evidence_note = (
            f"{len(evidence)} same-label BM25 examples were retrieved"
            if evidence
            else "no reliable same-label BM25 evidence was retrieved"
        )
        text_note = "The post contains claim-like social media language" if text else "The input text is empty"
        return (
            f"{text_note}. The hybrid system keeps the {decision_source} label "
            f"{prediction.label} ({label_name}) with confidence {prediction.confidence:.3f}; "
            f"{evidence_note}. LLM explanation generation failed, so this fallback rationale is used."
        )

    def predict(self, text: str) -> str:
        """
        Predict 0/1 and store a natural-language explanation.

        BM25 no longer votes on ordinary samples. It is filtered by the neural
        label and then used only to make the LLM explanation more grounded.
        """
        self.last_explanation = ""
        if not self.all_labels:
            return ""

        decision_source = "neural classifier"
        exact_label = self.retriever.get_exact_label(text) if self.use_exact_match else None

        if exact_label is not None:
            # Exact duplicates are deterministic and usually safer than model inference.
            # This shortcut is separate from BM25, so it does not inherit BM25 voting errors.
            prediction = NeuralPrediction(
                label=exact_label,
                confidence=1.0,
                probabilities={exact_label: 1.0},
            )
            decision_source = "exact training duplicate"
        else:
            prediction = self.classifier.predict(text)

        evidence = self.retriever.retrieve(
            text,
            top_k=self.evidence_top_k,
            preferred_label=prediction.label,
        )

        messages = self._make_explanation_messages(text, prediction, evidence, decision_source)
        try:
            response = self.call_llm(messages)
            explanation = self._parse_reasoning(response)
        except Exception:
            explanation = ""

        if not explanation:
            explanation = self._fallback_explanation(text, prediction, evidence, decision_source)

        self.last_explanation = explanation
        return prediction.label

    def get_last_explanation(self) -> str:
        """Return the explanation generated by the latest predict() call."""
        return self.last_explanation
