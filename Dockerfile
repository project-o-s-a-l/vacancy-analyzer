FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
WORKDIR /app
COPY requirements/linux-py312.txt /tmp/runtime.lock
RUN pip install --no-cache-dir -r /tmp/runtime.lock
COPY pyproject.toml README.md ./
COPY src ./src
COPY config/evaluation/benchmark.yaml config/evaluation/protocol.json config/evaluation/review.html ./config/evaluation/
COPY docs/evaluation/REVIEW.md ./docs/evaluation/REVIEW.md
RUN pip install --no-cache-dir --no-build-isolation --no-deps . && pip check
COPY alembic.ini ./
COPY alembic ./alembic
COPY config/profile.example.yaml ./config/profile.example.yaml
ENTRYPOINT ["vacancy-analyzer"]
CMD ["--help"]
