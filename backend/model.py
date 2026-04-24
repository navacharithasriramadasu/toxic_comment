"""
Toxicity Classifier using HuggingFace Transformers (BERT)
Falls back to a lightweight rule-based classifier if model loading fails.
"""

import os
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ToxicityClassifier:
    """
    BERT-based toxic comment classifier.
    Uses 'unitary/toxic-bert' pretrained model from HuggingFace — 
    fine-tuned on the Jigsaw Toxic Comment dataset.
    Falls back to rule-based scoring if torch/transformers unavailable.
    """

    TOXICITY_CATEGORIES = [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    ]

    CATEGORY_LABELS = {
        "toxic": "General Toxicity",
        "severe_toxic": "Severe Toxicity",
        "obscene": "Obscene Content",
        "threat": "Threatening Language",
        "insult": "Insulting Content",
        "identity_hate": "Identity-Based Hate",
    }

    def __init__(self, model_name: str = "unitary/toxic-bert"):
        self.model_name = model_name
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self._stats = {"total_analyzed": 0, "toxic_detected": 0, "blocked": 0}
        self._load_model()

    def _load_model(self):
        """Attempt to load BERT model; fall back to rule-based on failure."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            if torch.cuda.is_available():
                self.device = "cuda"
                self.model = self.model.cuda()
                logger.info("Using GPU acceleration")
            else:
                logger.info("Running on CPU")

            self.model.eval()
            logger.info("Model loaded successfully")
            self._use_bert = True

        except Exception as e:
            logger.warning(f"Could not load BERT model ({e}). Using rule-based fallback.")
            self._use_bert = False

    def predict(self, text: str) -> dict:
        """Run toxicity prediction on preprocessed text."""
        self._stats["total_analyzed"] += 1

        if self._use_bert and self.model and self.tokenizer:
            result = self._bert_predict(text)
        else:
            result = self._rule_based_predict(text)

        if result["is_toxic"]:
            self._stats["toxic_detected"] += 1

        return result

    def _bert_predict(self, text: str) -> dict:
        """BERT-based prediction using HuggingFace transformers."""
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        categories = {
            self.TOXICITY_CATEGORIES[i]: round(float(probs[i]), 4)
            for i in range(len(self.TOXICITY_CATEGORIES))
        }

        toxicity_score = float(probs[0])  # 'toxic' is index 0
        is_toxic = toxicity_score >= 0.5
        confidence = float(probs[0]) if is_toxic else float(1 - probs[0])

        return {
            "toxicity_score": toxicity_score,
            "is_toxic": is_toxic,
            "label": "TOXIC" if is_toxic else "NON-TOXIC",
            "confidence": confidence,
            "categories": categories,
        }

    def _rule_based_predict(self, text: str) -> dict:
        """
        Fallback rule-based toxicity scoring using keyword matching.
        Used when BERT model is not available.
        """
        toxic_patterns = {
            "severe_toxic": [
                "kill yourself", "kys", "die", "murder", "destroy you",
            ],
            "threat": [
                "i will hurt", "i'll kill", "you're dead", "watch your back",
            ],
            "obscene": [
                "fuck", "shit", "ass", "bitch", "cunt", "damn", "bastard",
            ],
            "insult": [
                "idiot", "stupid", "moron", "loser", "dumb", "ugly", "pathetic",
            ],
            "identity_hate": [
                "racist", "sexist", "homophobic", "slur",
            ],
            "toxic": [
                "hate", "terrible", "awful", "worst", "horrible", "disgust",
            ],
        }

        text_lower = text.lower()
        categories = {}
        max_score = 0.0

        for category, keywords in toxic_patterns.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            score = min(matches * 0.35, 1.0)
            categories[category] = round(score, 4)
            if score > max_score:
                max_score = score

        toxicity_score = min(max_score, 1.0)
        # Boost if multiple categories triggered
        triggered = sum(1 for v in categories.values() if v > 0)
        if triggered >= 2:
            toxicity_score = min(toxicity_score + 0.15, 1.0)

        is_toxic = toxicity_score >= 0.35
        confidence = toxicity_score if is_toxic else (1 - toxicity_score)

        return {
            "toxicity_score": round(toxicity_score, 4),
            "is_toxic": is_toxic,
            "label": "TOXIC" if is_toxic else "NON-TOXIC",
            "confidence": round(confidence, 4),
            "categories": categories,
        }

    def get_stats(self) -> dict:
        total = self._stats["total_analyzed"]
        toxic = self._stats["toxic_detected"]
        return {
            "total_analyzed": total,
            "toxic_detected": toxic,
            "non_toxic": total - toxic,
            "toxic_rate": round(toxic / total * 100, 2) if total > 0 else 0,
            "model": self.model_name,
            "mode": "bert" if self._use_bert else "rule-based",
        }
