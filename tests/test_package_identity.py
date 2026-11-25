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
        data = (root / name).read_bytes()
        target.write_bytes(
            data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") if name.endswith(".py") else data
        )
    snapshots = site_packages / "collectors/snapshots"
    snapshots.mkdir()
    (snapshots / "capture.html").write_text("local observation", encoding="utf-8")
    foreign = site_packages / "foreign_dependency"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("unrelated = True\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "__file__", str(site_packages / "jobs/evaluation.py"))
    assert evaluation.source_revision() == before
    (site_packages / "jobs/admission.py").write_text("changed = True\n", encoding="utf-8")
    assert evaluation.source_revision()["inventory_sha256"] != before["inventory_sha256"]
