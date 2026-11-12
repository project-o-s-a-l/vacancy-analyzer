from pathlib import Path

from jobs import evaluation


def test_wheel_layout_keeps_runtime_identity_and_excludes_foreign_dependencies(
    tmp_path, monkeypatch
):
    before = evaluation.source_revision()
    root = Path(evaluation.__file__).resolve().parents[2]
    site_packages = tmp_path / "Lib/site-packages"
    for name in before["files"]:
        target = site_packages / name.removeprefix("src/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    foreign = site_packages / "foreign_dependency"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("unrelated = True\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "__file__", str(site_packages / "jobs/evaluation.py"))
    assert evaluation.source_revision() == before
    (site_packages / "jobs/admission.py").write_text("changed = True\n", encoding="utf-8")
    assert evaluation.source_revision()["inventory_sha256"] != before["inventory_sha256"]
