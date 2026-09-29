# vacancy-analyzer
# Job Market Analyzer

Система для автоматического сбора, анализа и персонального ранжирования вакансий с нескольких job-платформ.

Проект предназначен для поиска наиболее подходящих вакансий на основе навыков, опыта и предпочтений кандидата. Система собирает новые вакансии, анализирует требования, сравнивает их с профилем пользователя, рассчитывает степень соответствия, а затем отправляет лучшие варианты в Telegram.

В дальнейшем проект будет использовать Data Science, NLP и Machine Learning для персонального ранжирования вакансий на основе пользовательского feedback.

---

## Основная идея

Вместо обычного агрегатора вакансий проект должен отвечать на вопрос:

> Какие из появившихся вакансий наиболее релевантны конкретному кандидату и почему?

Для каждой вакансии система должна определять:

* насколько навыки кандидата соответствуют требованиям;
* какие обязательные навыки присутствуют;
* каких навыков не хватает;
* соответствует ли опыт кандидата уровню вакансии;
* насколько вакансия семантически похожа на опыт и интересы кандидата;
* подходит ли зарплата, формат работы и местоположение;
* насколько вакансия интересна пользователю с учётом его предыдущего feedback.

После анализа пользователю отправляется краткая карточка вакансии в Telegram.

Пример:

```text
🔥 Match: 87%

Python Backend Developer
Example Company

Совпадает:
✅ Python
✅ FastAPI
✅ PostgreSQL
✅ Docker

Не хватает:
❌ Kafka
❌ Kubernetes

Причины рекомендации:
• сильное совпадение технического стека;
• подходящий уровень позиции;
• опыт пользователя соответствует требованиям.

[Открыть вакансию]

👍 Интересно
👎 Не подходит
📌 Сохранить
✉️ Сопроводительное
✅ Откликнулся
```

---

# Architecture

Общая архитектура проекта:

```text
HH / Habr Career / SuperJob / ...
              │
              ▼
        Data Collectors
              │
              ▼
        Raw vacancies
              │
              ▼
     Cleaning / Normalization
              │
       ┌──────┴──────┐
       ▼             ▼
 Vacancy features   Analytics
       │
       ▼
 Skill extraction / NLP
       │
       ▼
 Candidate ↔ Vacancy matching
       │
       ▼
     Ranking model
       │
       ▼
   Top vacancies
       │
       ├──────────────► Cover Letter Generator
       │
       ▼
      Telegram
       │
       ▼
 👍 / 👎 / Apply / Save / Skip reason
       │
       └──────────────► Training dataset
                              │
                              ▼
                         ML retraining
```

---

# Основные возможности

Планируемые возможности системы:

* сбор вакансий из нескольких источников;
* сохранение истории вакансий;
* нормализация данных из разных источников;
* удаление дубликатов;
* анализ рынка вакансий;
* извлечение технических навыков из описаний;
* построение единой taxonomy навыков;
* сопоставление вакансии с профилем кандидата;
* вычисление `match score`;
* semantic matching через text embeddings;
* персональное ML-ранжирование вакансий;
* анализ skill gap;
* выявление трендов рынка;
* генерация черновика сопроводительного письма;
* отправка рекомендаций через Telegram;
* сбор feedback пользователя;
* переобучение ranking model на пользовательском feedback.

---

# Источники вакансий

На первом этапе проект работает только с одним источником:

```text
HeadHunter
```

Позже планируется добавить:

```text
Habr Career
SuperJob
другие доступные источники
```

Каждый источник реализуется через общий интерфейс.

Пример:

```python
class VacancySource(ABC):

    @abstractmethod
    async def search(
        self,
        query: SearchQuery,
    ) -> list["Vacancy"]:
        ...

    @abstractmethod
    async def get_vacancy(
        self,
        vacancy_id: str,
    ) -> "Vacancy":
        ...
```

Бизнес-логика приложения не должна зависеть от конкретного API.

---

# Data Pipeline

Базовый pipeline обработки вакансии:

```text
External API
    │
    ▼
Raw JSON
    │
    ▼
Source Mapper
    │
    ▼
Domain Vacancy
    │
    ▼
PostgreSQL
    │
    ▼
Preprocessing
    │
    ▼
Skill Extraction
    │
    ▼
Feature Engineering
    │
    ▼
Scoring / ML
```

Исходный ответ внешнего API также должен сохраняться в базе в виде `raw_payload`.

Это позволяет повторно обработать исторические данные при изменении preprocessing pipeline.

---

# Domain Model

Все источники должны преобразовываться в единую внутреннюю модель вакансии.

Пример:

```python
class Vacancy:
    id: UUID

    source: str
    source_id: str
    url: str

    title: str
    company: str

    description: str | None
    skills: list[str]

    salary_from: int | None
    salary_to: int | None
    currency: str | None

    experience: str | None
    location: str | None
    remote: bool | None

    published_at: datetime
    collected_at: datetime
```

---

# Candidate Profile

Резюме пользователя и его профиль рассматриваются отдельно.

Структурированный профиль используется для matching и ML.

Пример:

```yaml
skills:

  python:
    level: advanced
    years: 3

  fastapi:
    level: intermediate
    years: 1.5

  postgresql:
    level: intermediate

  docker:
    level: intermediate

  pandas:
    level: beginner

  numpy:
    level: beginner

preferences:

  positions:
    - python developer
    - backend developer
    - data analyst
    - junior data scientist

  remote: preferred

  salary_min: null
```

Резюме хранится отдельно и используется в том числе для генерации сопроводительных писем.

---

# Data Science

Data Science — одна из центральных частей проекта.

Проект предполагает постепенное усложнение анализа: от классического EDA и rule-based matching до NLP и персонального ranking model.

---

## Market Analysis

После накопления достаточного количества вакансий планируется анализировать:

* количество вакансий во времени;
* распределение зарплат;
* наиболее востребованные навыки;
* навыки по уровням Junior / Middle / Senior;
* наиболее популярные комбинации технологий;
* различия требований между направлениями;
* изменение спроса на технологии;
* salary distributions;
* популярность удалённого формата;
* требования разных работодателей.

Пример анализа:

```text
Top Skills

Python        78%
SQL           65%
PostgreSQL    54%
Docker        48%
FastAPI       36%
Kafka         28%
Airflow       21%
```

---

## Skill Co-occurrence

Для анализа связей между технологиями планируется строить матрицы совместной встречаемости.

Пример:

```text
             Python Docker SQL Kafka Airflow

Python          -
Docker        0.71    -
SQL           0.82   0.54   -
Kafka         0.28   0.41  0.39   -
Airflow       0.22   0.30  0.45  0.31   -
```

Это позволит отвечать на вопросы:

* какие технологии обычно требуются вместе;
* какие навыки являются ядром определённого направления;
* какие технологии логично изучать следующими.

---

# Skill Extraction

На первом этапе skill extraction будет rule-based.

Пример taxonomy:

```python
SKILLS = {
    "python": [
        "python",
        "python3",
        "python 3",
    ],

    "postgresql": [
        "postgresql",
        "postgres",
        "psql",
    ],

    "scikit-learn": [
        "scikit-learn",
        "sklearn",
    ],

    "kubernetes": [
        "kubernetes",
        "k8s",
    ],
}
```

Skill taxonomy будет разделена по категориям:

```text
Programming Languages
Frameworks
Databases
DevOps
Cloud
Data Engineering
Data Science
Machine Learning
Testing
Architecture
```

Позже extraction pipeline может быть расширен NLP-моделями.

---

# Baseline Matching

До использования Machine Learning система должна иметь прозрачный baseline.

Для каждой вакансии рассчитывается несколько независимых score:

```text
skill_score
title_score
experience_score
salary_score
location_score
freshness_score
semantic_score
```

Пример итоговой формулы:

```text
TOTAL_SCORE =

0.40 × skill_score
+
0.20 × experience_score
+
0.15 × title_score
+
0.10 × salary_score
+
0.10 × preference_score
+
0.05 × freshness_score
```

Весовая модель не рассматривается как окончательное решение.

Она необходима как baseline для последующего сравнения с ML-моделью.

---

# Skill Matching

Навыки вакансии будут разделяться минимум на две категории:

```text
required
optional
```

Например:

```text
Vacancy

Required:
Python
FastAPI
PostgreSQL
Kafka

Optional:
Redis
Docker
```

Навыки могут иметь разные веса:

```text
required skill = 3
optional skill = 1
```

Базовая формула:

```python
skill_score = matched_weight / total_weight
```

---

# Semantic Matching

Одного keyword matching недостаточно.

Например:

```text
Resume:

"Разрабатывал высоконагруженные REST-сервисы"
```

и:

```text
Vacancy:

"Опыт проектирования backend API"
```

имеют похожий смысл, даже если содержат разные слова.

Для решения этой задачи планируется использовать text embeddings.

Pipeline:

```text
Resume / Candidate Experience
        │
        ▼
   Embedding Model
        │
        ▼
      Vector
        │
        │ cosine similarity
        ▼
      Vector
        ▲
        │
   Embedding Model
        ▲
        │
     Vacancy
```

При этом сравнивать полностью резюме и полностью вакансию не планируется.

Вакансия будет разделяться на:

```text
title
requirements
responsibilities
skills
```

Резюме:

```text
experience
projects
skills
education
```

После этого similarity будет рассчитываться между конкретными секциями.

---

# Retrieval & Reranking

На более позднем этапе pipeline может выглядеть следующим образом:

```text
Все новые вакансии
        │
        ▼
 Rules / Embeddings
        │
       TOP 100
        │
        ▼
     Reranker
        │
       TOP 20
        │
        ▼
Personal Ranking Model
        │
       TOP 5
        │
        ▼
     Telegram
```

Первый этап обеспечивает быстрый retrieval.

Второй — более точное ранжирование небольшого количества кандидатов.

---

# Machine Learning

После накопления пользовательского feedback появится возможность обучать персональную модель.

Целевая задача:

```text
P(vacancy is interesting | vacancy, candidate, context)
```

Примеры features:

```text
skill_coverage
required_skill_coverage
optional_skill_coverage

title_similarity
semantic_similarity

experience_difference

salary_difference

remote

vacancy_age

number_missing_required_skills

source

seniority
```

---

## Initial Models

Первая ML-модель:

```text
Logistic Regression
```

После неё планируется сравнить:

```text
Random Forest
Gradient Boosting
HistGradientBoosting
```

Сложные модели будут добавляться только в случае измеримого улучшения качества.

---

# User Feedback

Telegram используется не только как способ доставки вакансий, но и как источник training data.

Пользователь сможет отметить вакансию:

```text
👍 Интересно
👎 Не подходит
📌 Сохранить
✅ Откликнулся
```

Для негативного feedback можно дополнительно сохранить причину:

```text
salary
stack
seniority
location
company
responsibilities
other
```

---

# Labels

Вместо простого бинарного feedback планируется использовать graded relevance.

Например:

```text
0 — skip
1 — opened
2 — interesting
3 — saved
4 — applied
5 — interview
```

Это позволяет постепенно перейти от задачи classification к learning-to-rank.

---

# ML Metrics

Так как основная задача проекта — ranking, обычная Accuracy не является основной метрикой.

Основные метрики:

```text
Precision@5
Precision@10

Recall@5
Recall@10

NDCG@5
NDCG@10
```

Главный вопрос:

> Насколько хорошие вакансии находятся в верхней части рекомендаций?

---

# Validation

Так как вакансии являются временными данными, случайный `train_test_split` не должен использоваться как основной способ финальной проверки.

Предпочтительный вариант:

```text
September → January
        TRAIN

February
        TEST
```

То есть модель обучается на прошлом и проверяется на будущих данных.

Такой подход ближе к реальному production-сценарию.

---

# Skill Gap Analysis

Система должна не только рекомендовать вакансии, но и показывать, какие знания сильнее всего ограничивают кандидата.

Пример:

```text
Missing Skill       Suitable Vacancies

Docker                     151
SQL                        128
Kafka                       82
Airflow                     64
AWS                         54
Kubernetes                  49
```

На основе этого можно вычислять потенциальный эффект изучения конкретного навыка.

Например:

```text
Current skill coverage: 42%

After learning Docker: 57%

Potential improvement: +15 pp
```

---

# Market Trends

Исторические данные позволят анализировать изменение рынка.

Например:

```text
Kafka demand
      ▲
      │
      │             ╭────
      │         ╭───╯
      │     ╭───╯
      │ ╭───╯
      └────────────────────► time
```

Планируется отслеживать:

* рост и падение популярности технологий;
* новые часто встречающиеся технологии;
* изменение зарплат;
* изменение требований;
* изменение популярности различных специализаций.

---

# Cover Letter Generation

Генерация сопроводительного письма является последним этапом recommendation pipeline.

LLM не должна напрямую получать только:

```text
resume + vacancy
```

Предпочтительный pipeline:

```text
Vacancy
    │
    ▼
Requirements Extraction
    │
    ▼
Candidate Matching
    │
    ▼
Resume Evidence Retrieval
    │
    ▼
Structured Context
    │
    ▼
LLM
    │
    ▼
Cover Letter Draft
```

Система должна использовать только подтверждённые факты из резюме кандидата.

Она не должна придумывать:

* опыт работы;
* компании;
* проекты;
* технологии;
* достижения;
* количественные показатели.

---

# Telegram Bot

Telegram является основным пользовательским интерфейсом MVP.

Планируемые команды:

```text
/start
/status
/latest
/profile
/stats
```

Пример карточки:

```text
🔥 Match: 82%

Python Developer
Example Company

💰 180–250k
🌍 Remote
📊 Middle

Совпало:
✅ Python
✅ PostgreSQL
✅ FastAPI
✅ Docker

Не хватает:
❌ Kafka
❌ Kubernetes

Почему рекомендована:
• высокий skill match;
• подходящий уровень позиции;
• обязанности похожи на опыт пользователя.

[Открыть вакансию]

👍 Интересно
👎 Не подходит
📌 Сохранить
✉️ Сопроводительное
✅ Откликнулся
```

---

# Tech Stack

## Backend

```text
Python
Pydantic
SQLAlchemy
Alembic
httpx
asyncio
```

## Database

```text
PostgreSQL
```

В будущем:

```text
pgvector
```

## Data Science

```text
NumPy
pandas
scikit-learn
Jupyter
```

## NLP

```text
sentence-transformers
text embeddings
reranking models
```

## Telegram

```text
aiogram
```

## Infrastructure

```text
Docker
Docker Compose
```

## Code Quality

```text
pytest
ruff
```

---

# Local Development

Проект поддерживает Python 3.12 и новее.

Создание и активация виртуального окружения в PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --group dev
```

Ручной запуск проверок:

```powershell
ruff check .
ruff format --check .
pytest
```

Ruff и pytest настраиваются централизованно в `pyproject.toml`. Проверки запускаются
разработчиком локально перед созданием коммита и Pull Request.

Подробный командный процесс разработки, создания коммитов и подготовки Pull Request описан в
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

---

# Project Structure

Планируемая структура проекта:

```text
job-market-analyzer/

├── src/
│   │
│   ├── collectors/
│   │   ├── base.py
│   │   ├── hh.py
│   │   ├── habr.py
│   │   └── superjob.py
│   │
│   ├── domain/
│   │   ├── vacancy.py
│   │   ├── candidate.py
│   │   └── feedback.py
│   │
│   ├── storage/
│   │   ├── database.py
│   │   ├── models/
│   │   └── repositories/
│   │
│   ├── processing/
│   │   ├── cleaner.py
│   │   ├── deduplicator.py
│   │   └── normalizer.py
│   │
│   ├── nlp/
│   │   ├── skills.py
│   │   ├── embeddings.py
│   │   └── extraction.py
│   │
│   ├── scoring/
│   │   ├── baseline.py
│   │   ├── skill_score.py
│   │   └── semantic_score.py
│   │
│   ├── ml/
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── letters/
│   │   ├── retrieval.py
│   │   └── generator.py
│   │
│   ├── bot/
│   │   ├── handlers.py
│   │   └── keyboards.py
│   │
│   └── jobs/
│       ├── collect.py
│       ├── rank.py
│       └── notify.py
│
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_market_eda.ipynb
│   ├── 03_skills.ipynb
│   ├── 04_matching.ipynb
│   └── 05_ml_model.ipynb
│
├── tests/
│
├── docs/
│
├── docker/
│
├── alembic/
│
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# Development Principles

## 1. Start Simple

Для каждой сложной задачи сначала реализуется прозрачный baseline.

Например:

```text
Skill Matching

keyword matching
        ↓
embeddings
        ↓
ML ranking
```

---

## 2. Measure Before Replacing

ML-модель должна заменять baseline только тогда, когда есть измеримое улучшение качества.

---

## 3. Raw Data Must Be Preserved

Необработанные данные источников должны сохраняться.

Это позволяет повторно запускать preprocessing без повторного обращения к API.

---

## 4. Business Logic Must Be Source Independent

Ни scoring, ни ML, ни Telegram не должны зависеть от структуры конкретного job-сайта.

---

## 5. No Hallucinations in Cover Letters

Система не должна добавлять в сопроводительное письмо факты, отсутствующие в резюме или профиле пользователя.

---

## 6. Notebooks Are for Research

Исследования и эксперименты выполняются в notebooks.

Production-код переносится в:

```text
src/
```

---

## 7. ML Must Be Reproducible

Эксперименты должны фиксировать:

* dataset;
* preprocessing;
* features;
* model;
* hyperparameters;
* metrics;
* train/test period.

---

# Project Status

Текущий этап:

```text
[ ] Project setup
[ ] PostgreSQL
[ ] Vacancy domain model
[ ] HH integration
[ ] Vacancy storage
[ ] Data cleaning
[ ] Market EDA
[ ] Skill extraction
[ ] Candidate profile
[ ] Baseline matching
[ ] Telegram MVP
[ ] Feedback collection
[ ] Embeddings
[ ] ML ranking
[ ] Skill-gap analysis
[ ] Market trends
[ ] Cover letter generation
```

---

# Long-Term Goal

Финальная система должна автоматически:

```text
Collect
   ↓
Normalize
   ↓
Analyze
   ↓
Extract Skills
   ↓
Match
   ↓
Rank
   ↓
Explain
   ↓
Notify
   ↓
Collect Feedback
   ↓
Learn
```

Основная цель проекта — построить не просто агрегатор вакансий, а персональную Data Science / Machine Learning систему, которая постепенно адаптируется к кандидату и помогает принимать решения о поиске работы на основе реальных данных рынка.
