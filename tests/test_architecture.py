import ast
import subprocess
import sys
from pathlib import Path

import pytest

from jobs.cli import parser


@pytest.mark.parametrize(
    "relative",
    [
        "jobs/users.py",
        "jobs/profiles.py",
        "jobs/collect.py",
        "jobs/collection_scheduler.py",
        "jobs/collection_runtime.py",
        "jobs/vacancy_analysis.py",
        "nlp/ollama.py",
        "jobs/notify.py",
        "jobs/resumes.py",
        "jobs/rank.py",
        "jobs/matching_dataset.py",
        "jobs/matching_labels.py",
        "jobs/matching_audit.py",
        "jobs/ranker_training.py",
        "scoring/text_features.py",
        "scoring/linear_ranker.py",
        "scoring/matching_evaluation.py",
        "scoring/baseline.py",
        "domain/matching.py",
        "domain/matching_dataset.py",
        "domain/ranker.py",
    ],
)
def test_business_logic_has_no_telegram_dependencies(relative):
    path = Path(__file__).parents[1] / "src" / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    assert not any(name.split(".")[0] in {"bot", "aiogram"} for name in modules)


def test_market_and_chart_commands_are_removed():
    choices = parser()._subparsers._group_actions[0].choices
    assert "market" not in choices
    with pytest.raises(SystemExit):
        parser().parse_args(["market", "--html", "report.html"])


def test_normal_cli_does_not_load_ml_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from jobs.cli import parser; parser(); "
            "assert 'sklearn' not in sys.modules; assert 'scipy' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
