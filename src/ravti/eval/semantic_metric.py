from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SemanticSimilarityMetric:
    """
    Trait-level semantic comparison (BioCAP-style).

    Production path: call GPT-4o / InternVL on paired images, then compare text.
    This implementation optionally uses BERTScore when `reference_text` and
    `candidate_text` are provided; otherwise returns neutral 0.5.
    """

    lang: str = "en"

    def compare_texts(self, reference: str, candidate: str) -> float:
        if not reference.strip() or not candidate.strip():
            return 0.5
        try:
            from bert_score import score as bert_score  # type: ignore

            p, r, f1 = bert_score([candidate], [reference], lang=self.lang, rescale_with_baseline=True)
            return float(f1.mean())
        except Exception:
            return 0.5

    def compare_from_llm_descriptions(
        self,
        reference_description: Optional[str],
        candidate_description: Optional[str],
    ) -> float:
        if reference_description is None or candidate_description is None:
            return 0.5
        return self.compare_texts(reference_description, candidate_description)
