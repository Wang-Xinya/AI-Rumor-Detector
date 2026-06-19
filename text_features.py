import re
import string
import numpy as np


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")


def clean_text(text: str) -> str:
    """
    Normalize tweet-like text for TF-IDF features.
    """
    text = "" if text is None else str(text)

    text = URL_PATTERN.sub(" urltoken ", text)
    text = USER_PATTERN.sub(" usertoken ", text)
    text = HASHTAG_PATTERN.sub(r" hashtagtoken \1 ", text)
    text = re.sub(r"\brt\b", " ", text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_stats(texts):
    """
    Extract simple non-negative style/statistical features from raw text.
    These features are useful for rumor detection on tweet-like data.
    """
    features = []

    for text in texts:
        raw = "" if text is None else str(text)
        words = raw.split()

        length = len(raw)
        word_count = len(words)

        url_count = len(URL_PATTERN.findall(raw))
        user_count = len(USER_PATTERN.findall(raw))
        hashtag_count = len(HASHTAG_PATTERN.findall(raw))

        exclamation_count = raw.count("!")
        question_count = raw.count("?")
        digit_count = sum(ch.isdigit() for ch in raw)

        alpha_count = sum(ch.isalpha() for ch in raw)
        upper_count = sum(ch.isupper() for ch in raw)
        uppercase_ratio = upper_count / max(alpha_count, 1)

        punct_count = sum(ch in string.punctuation for ch in raw)
        punct_ratio = punct_count / max(length, 1)

        avg_word_len = (
            sum(len(w) for w in words) / max(word_count, 1)
        )

        repeated_punct = len(re.findall(r"([!?])\1+", raw))

        features.append(
            [
                length,
                word_count,
                url_count,
                user_count,
                hashtag_count,
                exclamation_count,
                question_count,
                digit_count,
                uppercase_ratio,
                punct_ratio,
                avg_word_len,
                repeated_punct,
            ]
        )

    return np.asarray(features, dtype=float)