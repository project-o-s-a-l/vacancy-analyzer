"""Corpus-fitted lexical features, without pretrained weights or relevance labels."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version

from nlp.text import word_tokens
from processing.cleaner import clean_html

TEXT_FEATURES_VERSION = "tfidf-v1"
FIXED_TEXT_FEATURES_VERSION = "fixed-cosine-v1"
WORD_LIMIT = 30_000
CHAR_LIMIT = 50_000


class TextFeaturesUnavailable(ValueError):
    pass


def text_tokens(text: str) -> tuple[str, ...]:
    # Keep one-letter technologies (C, R), punctuation suffixes (C++, C#), and negations.
    return tuple(token for token in word_tokens(text) if any(char.isalpha() for char in token))


def normalized_document(text: str) -> str:
    return " ".join(text_tokens(clean_html(text)))


@dataclass(frozen=True)
class TextComparison:
    word_cosine: float | None
    char_cosine: float | None


class CorpusTextIndex:
    """Fit only on job texts. Query texts never change vocabulary or IDF."""

    def __init__(self, documents: list[str]) -> None:
        if not documents:
            raise ValueError("Для текстового сравнения нужны вакансии.")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise TextFeaturesUnavailable(
                'Установите зависимости эксперимента: python -m pip install -e ".[ml]"'
            ) from None

        self.documents = [normalized_document(text) for text in documents]
        self.spec = {
            "version": TEXT_FEATURES_VERSION,
            "sklearn": version("scikit-learn"),
            "pymorphy3": version("pymorphy3"),
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "word_max_features": WORD_LIMIT,
            "char_max_features": CHAR_LIMIT,
            "sublinear_tf": True,
            "smooth_idf": True,
            "norm": "l2",
            "fit_scope": "eligible_active_vacancies_only",
            "corpus_text_hash": sha256("\n\0\n".join(self.documents).encode()).hexdigest(),
        }
        self.word = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=WORD_LIMIT,
            sublinear_tf=True,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=False,
            ngram_range=(3, 5),
            max_features=CHAR_LIMIT,
            sublinear_tf=True,
        )
        self.word_matrix = self.word.fit_transform(self.documents) if any(self.documents) else None
        self.char_matrix = self.char.fit_transform(self.documents) if any(self.documents) else None

    def compare(self, text: str) -> tuple[list[TextComparison], float]:
        document = normalized_document(text)
        tokens = document.split()
        known = (
            sum(token in self.word.vocabulary_ for token in tokens) / len(tokens)
            if tokens and self.word_matrix is not None
            else 0.0
        )
        channels = []
        for vectorizer, matrix in ((self.word, self.word_matrix), (self.char, self.char_matrix)):
            scores: list[float | None] = [None] * len(self.documents)
            if matrix is not None and document:
                query = vectorizer.transform([document])
                if query.nnz:
                    # L2-normalized vectors: sparse dot product equals cosine similarity.
                    values = (matrix @ query.T).toarray().ravel()
                    scores = [
                        round(max(0.0, min(1.0, float(value))), 8) if matrix[i].nnz else None
                        for i, value in enumerate(values)
                    ]
            channels.append(scores)
        return [TextComparison(*values) for values in zip(*channels, strict=True)], round(known, 8)
