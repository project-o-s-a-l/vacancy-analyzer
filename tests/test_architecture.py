import ast
import subprocess
import sys
from pathlib import Path

import pytest

from jobs.cli import parser


def test_business_logic_has_no_telegram_dependencies():
    src = Path(__file__).parents[1] / "src"
    cli_adapters = {src / "jobs/cli.py", src / "jobs/resume_cli.py"}
    violations = []
    for package in ("domain", "jobs", "nlp", "scoring"):
        for path in sorted((src / package).rglob("*.py")):
            if path in cli_adapters:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    if module.split(".")[0] in {"bot", "aiogram"}:
                        violations.append(f"{path.relative_to(src)}:{node.lineno} imports {module}")
    assert not violations, violations


def test_retired_cli_commands_are_rejected():
    cli = parser()
    for command in ("market", "demo", "telegram-preview"):
        with pytest.raises(SystemExit) as error:
            cli.parse_args([command])
        assert error.value.code == 2, command


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
