"""Score labels are UI, deliberately outside model and admission contracts."""


def score_label(match):
    if match.ranking and match.ranking.applied:
        return f"Индекс релевантности: {match.ranking.model_score:.3f} (модель)"
    return f"Соответствие: {match.score:.1f}% (по правилам)"


def evidence_label(match):
    if match.ranking and match.ranking.applied:
        return f"Известные профессиональные сведения: {match.ranking.evidence_support:.0%}"
    return f"Полнота сравнения: {match.evidence_coverage:.0%}"
