"""Watchdog around an unloaded-after-use model process; no service or GPU allocation."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_RSS = 1024 * 1024 * 1024
MIN_AVAILABLE = 4 * 1024 * 1024 * 1024
MAX_SECONDS = 600


def encode_bounded(tasks, model_directory):
    if not tasks:
        return _encode_bounded(tasks, model_directory)
    from nlp import ollama_slot

    # Same host slot as the existing Ollama adapters; don't pile model work onto the PC.
    with ollama_slot.LOCK_PATH.open("a+b") as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b"0")
            stream.flush()
        if not ollama_slot._lock(stream):
            raise ValueError("Слот модели занят. Повторите эксперимент после текущего разбора")
        try:
            _release_retained_ollama()
            return _encode_bounded(tasks, model_directory)
        finally:
            ollama_slot._unlock(stream)


def _release_retained_ollama():
    """The file slot serializes native work; retained weights must also be released."""
    import httpx

    from jobs.settings import Settings
    from nlp import ollama_slot

    if ollama_slot.LOCK_PATH.with_suffix(".uncertain.json").exists():
        raise ValueError("Предыдущий запрос модели не завершён; дождитесь восстановления worker")
    with httpx.Client(base_url=Settings().ollama_base_url, timeout=10, trust_env=False) as client:
        try:
            response = client.get("/api/ps", timeout=2)
        except httpx.ConnectError:
            return  # No local service is running; ONNX still has its own watchdog.
        response.raise_for_status()
        for loaded in response.json()["models"]:
            reply = client.post(
                "/api/generate", json={"model": loaded["name"], "keep_alive": 0, "stream": False}
            )
            reply.raise_for_status()
            if not reply.json().get("done"):
                raise ValueError("Не подтверждено освобождение модели")
        deadline = time.monotonic() + 10
        while True:
            state = client.get("/api/ps", timeout=2)
            state.raise_for_status()
            if not state.json()["models"]:
                break
            if time.monotonic() >= deadline:
                raise ValueError("Модель ещё занята; семантический эксперимент отложен")
            time.sleep(0.1)


def _encode_bounded(tasks, model_directory):
    if not tasks:
        return {"values": [], "peak_rss_bytes": 0, "elapsed_seconds": 0, "load_seconds": 0}
    try:
        import psutil
    except ImportError as exc:
        raise ValueError("Установите дополнение .[semantic] для локального эксперимента") from exc

    if psutil.virtual_memory().available < MIN_AVAILABLE:
        raise ValueError("Для безопасного эксперимента нужно не менее 4 ГиБ свободной RAM")
    workspace = Path(".local-tmp")
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semantic-", dir=workspace) as directory:
        input_path, output_path = Path(directory) / "input.json", Path(directory) / "output.json"
        input_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        environment = os.environ.copy() | {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "jobs.semantic_worker",
                str(input_path),
                str(output_path),
                str(model_directory),
            ],
            env=environment,
            stdin=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        child, peak, started = psutil.Process(process.pid), 0, time.monotonic()
        try:
            affinity = child.cpu_affinity()[:1] if hasattr(child, "cpu_affinity") else None

            def sample_tree():
                # Windows venv python.exe is a launcher: monitor and limit its child too.
                resident = 0
                for member in [child, *child.children(recursive=True)]:
                    try:
                        member.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
                        if affinity:
                            member.cpu_affinity(affinity)
                        resident += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                return resident

            peak = sample_tree()
            process.stdin.write(b"1")
            process.stdin.close()
            while process.poll() is None:
                try:
                    peak = max(peak, sample_tree())
                except psutil.NoSuchProcess:
                    break
                if (
                    peak > MAX_RSS
                    or psutil.virtual_memory().available < MIN_AVAILABLE
                    or time.monotonic() - started > MAX_SECONDS
                ):
                    raise ValueError("Эксперимент остановлен по лимиту памяти или времени")
                time.sleep(0.1)
            if process.wait() != 0 or not output_path.is_file():
                raise ValueError("Локальное вычисление представлений завершилось с ошибкой")
            result = json.loads(output_path.read_text(encoding="utf-8"))
            result["peak_rss_bytes"] = peak
            return result
        finally:
            if process.poll() is None:
                for member in reversed(child.children(recursive=True)):
                    try:
                        member.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
