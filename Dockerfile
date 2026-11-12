FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config/evaluation/protocol.json config/evaluation/review.html ./config/evaluation/
COPY docs/evaluation/REVIEW.md ./docs/evaluation/REVIEW.md
RUN pip install --no-cache-dir .
COPY alembic.ini ./
COPY alembic ./alembic
COPY config/profile.example.yaml ./config/profile.example.yaml
ENTRYPOINT ["vacancy-analyzer"]
CMD ["--help"]
