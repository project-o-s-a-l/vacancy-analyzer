import sys

import pytest

from scoring.text_features import CorpusTextIndex, TextFeaturesUnavailable

pytest.importorskip("sklearn", reason='Install the optional ".[ml]" dependencies')


@pytest.mark.parametrize(
    "query,positive,negative",
    [
        ("Java Spring", "Java Spring SQL", "JavaScript React"),
        ("ведение бухгалтерского учёта", "ведение бухгалтерского учета", "ремонт автомобилей"),
        ("настройка роялей", "настройки рояля", "сварка металлов"),
        ("приём пациентов", "приема пациентов", "реставрация мебели"),
        ("перевозка грузов", "перевозки груза", "проведение переговоров"),
        ("приготовление блюд", "приготовления блюд", "ведение учета"),
        ("ведение переговоров", "ведение переговоров", "ремонт помещений"),
        ("C++", "C++", "C#"),
    ],
)
def test_word_features_work_across_professions(query, positive, negative):
    index = CorpusTextIndex([positive, negative])
    matches, known = index.compare(query)
    assert matches[0].word_cosine > matches[1].word_cosine
    assert known == 1


def test_exact_words_do_not_confuse_java_with_javascript():
    matches, _ = CorpusTextIndex(["Java", "JavaScript"]).compare("Java")
    assert matches[0].word_cosine == 1 and matches[1].word_cosine == 0
    # A separate character signal is deliberately not treated as skill equivalence.
    assert matches[1].char_cosine > 0


def test_query_does_not_fit_vocabulary_or_change_corpus_statistics():
    index = CorpusTextIndex(["Java Spring", "Excel учет"])
    vocabulary = dict(index.word.vocabulary_)
    weights = index.word.idf_.copy()
    matches, known = index.compare("qzxwvu")
    assert all(match.word_cosine is None for match in matches)
    assert all(match.char_cosine is None for match in matches)
    assert known == 0
    assert index.word.vocabulary_ == vocabulary
    assert (index.word.idf_ == weights).all()


def test_empty_documents_and_queries_are_unknown_instead_of_matches():
    index = CorpusTextIndex(["Java", "", "123 !!!"])
    matches, _ = index.compare("Java")
    assert matches[0].word_cosine == 1
    assert matches[1].word_cosine is None and matches[2].char_cosine is None
    assert all(match.word_cosine is None for match in index.compare("")[0])
    assert all(match.word_cosine is None for match in CorpusTextIndex(["", "123"]).compare("x")[0])
    with pytest.raises(ValueError):
        CorpusTextIndex([])


def test_html_scripts_do_not_become_text_evidence():
    matches, _ = CorpusTextIndex(["<script>secret</script><p>Java</p>", "Python"]).compare("secret")
    assert all(match.word_cosine is None for match in matches)


def test_dependency_failure_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    with pytest.raises(TextFeaturesUnavailable, match="pip install"):
        CorpusTextIndex(["Java"])
