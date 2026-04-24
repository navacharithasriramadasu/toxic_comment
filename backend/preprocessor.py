"""
Text Preprocessor for Toxic Comment Detection
Uses NLTK for NLP preprocessing pipeline.
"""

import re
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    NLP preprocessing pipeline:
    1. Lowercase
    2. URL/email removal
    3. HTML tag stripping
    4. Special char normalization
    5. Whitespace normalization
    Note: We keep punctuation and capitalization cues for BERT — it handles
    its own tokenization. Full stemming/lemmatization is applied for rule-based mode.
    """

    def __init__(self):
        self._nltk_ready = False
        self._init_nltk()

    def _init_nltk(self):
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            self._nltk_ready = True
            logger.info("NLTK initialized")
        except Exception as e:
            logger.warning(f"NLTK not available: {e}")

    def clean(self, text: str) -> str:
        """
        Apply preprocessing pipeline to raw comment text.
        Preserves semantic cues for transformer models.
        """
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "[URL]", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)

        # Normalize repeated characters (e.g., "stooopid" → "stoopid")
        text = re.sub(r"(.)\1{3,}", r"\1\1", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Limit length to 512 tokens worth (~2000 chars)
        if len(text) > 2000:
            text = text[:2000]

        return text

    def tokenize(self, text: str) -> list:
        """Tokenize text into words using NLTK if available."""
        if self._nltk_ready:
            try:
                from nltk.tokenize import word_tokenize
                return word_tokenize(text.lower())
            except Exception:
                pass
        return text.lower().split()

    def remove_stopwords(self, tokens: list) -> list:
        """Remove English stopwords from token list."""
        if self._nltk_ready:
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words("english"))
                return [t for t in tokens if t not in stop_words]
            except Exception:
                pass
        return tokens
