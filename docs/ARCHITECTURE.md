# Job Market Analyzer — Architecture

## 1. Назначение документа

Этот документ описывает архитектуру проекта **Job Market Analyzer**, структуру каталогов, назначение основных модулей, направление зависимостей и общий поток обработки данных.

Job Market Analyzer — система для автоматического сбора, анализа и персонального ранжирования вакансий. Система получает вакансии из внешних источников, нормализует данные, извлекает признаки, сравнивает вакансии с профилем кандидата и отправляет наиболее релевантные варианты через Telegram.

В дальнейшем проект предполагает использование NLP и Machine Learning для персонального ранжирования вакансий на основе пользовательского feedback.

---

# 2. Архитектурный подход

Проект строится как **модульный Python-монолит с Data/ML pipeline**.

Основные части приложения разделены по ответственности:

```text
Collectors
    ↓
Domain
    ↓
Processing
    ↓
Storage
    ↓
NLP
    ↓
Scoring
    ↓
Machine Learning
    ↓
Ranking
    ↓
Telegram
    ↓
Feedback
```

На текущем этапе проект не разделяется на микросервисы.

Это позволяет:

* сохранить простую структуру проекта;
* упростить локальную разработку;
* уменьшить количество инфраструктурных зависимостей;
* избежать лишнего сетевого взаимодействия;
* сохранить чёткие границы между модулями;
* при необходимости в дальнейшем вынести отдельные компоненты в самостоятельные сервисы.

---

# 3. Общая структура проекта

```text
job-market-analyzer/
│
├── src/
│   │
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hh.py
│   │   ├── habr.py
│   │   └── superjob.py
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── vacancy.py
│   │   ├── candidate.py
│   │   └── feedback.py
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── vacancy.py
│   │   │   ├── candidate.py
│   │   │   └── feedback.py
│   │   │
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── vacancy_repository.py
│   │       ├── candidate_repository.py
│   │       └── feedback_repository.py
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── normalizer.py
│   │   └── deduplicator.py
│   │
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── skills.py
│   │   ├── extraction.py
│   │   └── embeddings.py
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── skill_score.py
│   │   └── semantic_score.py
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── letters/
│   │   ├── __init__.py
│   │   ├── retrieval.py
│   │   └── generator.py
│   │
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── handlers.py
│   │   └── keyboards.py
│   │
│   └── jobs/
│       ├── __init__.py
│       ├── collect.py
│       ├── rank.py
│       └── notify.py
│
├── tests/
│   ├── collectors/
│   ├── storage/
│   ├── processing/
│   ├── nlp/
│   ├── scoring/
│   ├── ml/
│   ├── bot/
│   └── jobs/
│
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_market_eda.ipynb
│   ├── 03_skills.ipynb
│   ├── 04_matching.ipynb
│   └── 05_ml_model.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── artifacts/
│   ├── ml_models/
│   ├── embeddings/
│   ├── metrics/
│   └── reports/
│
├── alembic/
│   ├── versions/
│   └── env.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── GIT_GUIDELINES.md
│
├── docker/
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── alembic.ini
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

Базовая модульная структура `src/`, `notebooks/`, `tests/`, `docs/`, `docker/` и `alembic/` уже заложена в проекте.

---

# 4. `src/`

```text
src/
```

Главный каталог production-кода проекта.

Здесь располагается код, который используется непосредственно приложением.

В `src/` не должны находиться:

* Jupyter notebooks;
* локальные датасеты;
* обученные бинарные ML-модели;
* временные файлы;
* результаты экспериментов.

Исследовательский код после проверки должен переноситься из `notebooks/` в соответствующий модуль `src/`.

---

# 5. `src/collectors/`

```text
src/collectors/
```

Отвечает за получение вакансий из внешних источников.

Примеры источников:

```text
HeadHunter
Habr Career
SuperJob
```

На первом этапе основным источником является HeadHunter.

## Структура

```text
collectors/
├── base.py
├── hh.py
├── habr.py
└── superjob.py
```

### `base.py`

Содержит общий интерфейс источника вакансий.

Например:

```python
class VacancySource(ABC):

    @abstractmethod
    async def search(self, query):
        ...

    @abstractmethod
    async def get_vacancy(self, vacancy_id):
        ...
```

Все collectors должны предоставлять единый интерфейс.

### `hh.py`

Реализация взаимодействия с HeadHunter.

### `habr.py`

В будущем — интеграция с Habr Career.

### `superjob.py`

В будущем — интеграция с SuperJob.

## Ограничение ответственности

Collector должен знать структуру API конкретного источника.

Остальная система — не должна.

```text
HeadHunter JSON
      ↓
HH Collector
      ↓
Domain Vacancy
```

Бизнес-логика приложения должна оставаться независимой от конкретного API.

---

# 6. `src/domain/`

```text
src/domain/
```

Содержит основные сущности предметной области.

```text
domain/
├── vacancy.py
├── candidate.py
└── feedback.py
```

## `vacancy.py`

Внутренняя модель вакансии.

Например:

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
```

Внешние источники должны преобразовывать данные к единой внутренней модели вакансии.

## `candidate.py`

Содержит профиль кандидата:

```text
skills
experience
preferred positions
salary preferences
remote preferences
location
```

## `feedback.py`

Описывает действия пользователя:

```text
interesting
not suitable
saved
applied
opened
interview
```

## Важно

`domain` не должен зависеть:

* от HeadHunter;
* от Telegram;
* от SQLAlchemy;
* от конкретной ML-модели.

---

# 7. `src/storage/`

```text
src/storage/
```

Отвечает за постоянное хранение данных.

Основная база данных проекта:

```text
PostgreSQL
```

В дальнейшем возможно использование `pgvector`.

Структура:

```text
storage/
├── database.py
├── models/
└── repositories/
```

---

# 8. `src/storage/database.py`

Содержит инфраструктуру подключения к PostgreSQL.

Например:

```text
SQLAlchemy Engine
SessionFactory
database configuration
connection management
```

В этом файле не должна находиться бизнес-логика.

---

# 9. `src/storage/models/`

```text
src/storage/models/
```

Содержит **ORM-модели SQLAlchemy**.

Это модели представления данных внутри PostgreSQL.

Например:

```text
models/
├── vacancy.py
├── candidate.py
└── feedback.py
```

Пример:

```python
class VacancyModel(Base):
    __tablename__ = "vacancies"

    ...
```

## Почему существует одновременно `domain/vacancy.py`

`domain.Vacancy` и `storage.models.VacancyModel` имеют разные задачи.

```text
Domain Model
     │
     │ описывает бизнес-сущность
     ▼

Vacancy


SQLAlchemy Model
     │
     │ описывает таблицу БД
     ▼

VacancyModel
```

Таким образом:

```text
src/domain/          → модели предметной области
src/storage/models/  → модели таблиц PostgreSQL
```

---

# 10. `src/storage/repositories/`

```text
src/storage/repositories/
```

Слой доступа к данным.

Например:

```text
vacancy_repository.py
candidate_repository.py
feedback_repository.py
```

Repository скрывает детали SQLAlchemy от основной логики.

Вместо:

```python
session.query(...)
```

основной код должен по возможности использовать:

```python
vacancy_repository.get_by_id(...)
vacancy_repository.save(...)
vacancy_repository.get_new(...)
```

Пример потока:

```text
Application Logic
       ↓
Repository
       ↓
SQLAlchemy
       ↓
PostgreSQL
```

---

# 11. `src/processing/`

```text
src/processing/
```

Отвечает за подготовку данных.

Структура:

```text
processing/
├── cleaner.py
├── normalizer.py
└── deduplicator.py
```

## `cleaner.py`

Очистка входящих данных.

Например:

* удаление лишних пробелов;
* очистка HTML;
* обработка пустых значений;
* исправление некорректных полей.

## `normalizer.py`

Приведение данных из разных источников к общему формату.

Например:

```text
"Москва"
"Moscow"
"г. Москва"

        ↓

единое значение
```

## `deduplicator.py`

Определяет повторяющиеся вакансии.

Например:

```text
одинаковый source_id
одинаковый URL
одинаковая вакансия из нескольких источников
```

---

# 12. `src/nlp/`

```text
src/nlp/
```

Отвечает за анализ текста.

```text
nlp/
├── skills.py
├── extraction.py
└── embeddings.py
```

## `skills.py`

Содержит taxonomy навыков.

Например:

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
}
```

## `extraction.py`

Извлекает навыки из текста вакансии.

На первом этапе используется rule-based подход.

## `embeddings.py`

Отвечает за получение vector embeddings текста.

Используется для semantic matching вакансии и опыта кандидата.

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

---

# 13. `src/scoring/`

```text
src/scoring/
```

Содержит прозрачную baseline-систему оценки вакансий.

```text
scoring/
├── baseline.py
├── skill_score.py
└── semantic_score.py
```

## `skill_score.py`

Рассчитывает совпадение навыков.

Например:

```text
Required:
Python
FastAPI
PostgreSQL

Matched:
Python
PostgreSQL
```

## `semantic_score.py`

Рассчитывает semantic similarity между вакансией и профилем кандидата.

## `baseline.py`

Объединяет несколько score в итоговый baseline.

Например:

```text
skill_score
experience_score
title_score
salary_score
preference_score
freshness_score
semantic_score
```

Baseline необходим как простой и объяснимый алгоритм до использования ML.

---

# 14. `src/ml/`

```text
src/ml/
```

Содержит **код работы с Machine Learning**.

Важно:

```text
src/ml/
```

содержит Python-код ML.

Он **не содержит сами обученные модели**.

Структура:

```text
ml/
├── features.py
├── train.py
├── predict.py
└── evaluate.py
```

## `features.py`

Формирует признаки для ML.

Например:

```text
skill_coverage
required_skill_coverage
semantic_similarity
experience_difference
salary_difference
vacancy_age
seniority
```

## `train.py`

Обучает модели.

Первая предполагаемая модель:

```text
Logistic Regression
```

В дальнейшем:

```text
Random Forest
Gradient Boosting
HistGradientBoosting
```

## `predict.py`

Использует обученную модель для получения prediction или ranking score.

## `evaluate.py`

Отвечает за оценку качества.

Основные метрики проекта:

```text
Precision@5
Precision@10
Recall@5
Recall@10
NDCG@5
NDCG@10
```

---

# 15. `src/letters/`

```text
src/letters/
```

Отвечает за генерацию черновиков сопроводительных писем.

Структура:

```text
letters/
├── retrieval.py
└── generator.py
```

## `retrieval.py`

Выбирает подтверждённые факты из профиля или резюме кандидата.

## `generator.py`

Формирует запрос к LLM и получает черновик письма.

Pipeline:

```text
Vacancy
    ↓
Requirements Extraction
    ↓
Candidate Matching
    ↓
Resume Evidence Retrieval
    ↓
Structured Context
    ↓
LLM
    ↓
Cover Letter Draft
```

Система не должна добавлять факты, отсутствующие в профиле или резюме пользователя.

---

# 16. `src/bot/`

```text
src/bot/
```

Telegram-интерфейс приложения.

Структура:

```text
bot/
├── handlers.py
└── keyboards.py
```

## `handlers.py`

Обрабатывает команды и действия пользователя.

Например:

```text
/start
/status
/latest
/profile
/stats
```

## `keyboards.py`

Содержит Telegram-клавиатуры и кнопки.

Например:

```text
👍 Интересно
👎 Не подходит
📌 Сохранить
✉️ Сопроводительное
✅ Откликнулся
```

Telegram является основным интерфейсом MVP.

## Ограничение

`bot` не должен реализовывать:

```text
skill extraction
scoring
ML training
database queries напрямую
```

Он вызывает соответствующие компоненты приложения.

---

# 17. `src/jobs/`

```text
src/jobs/
```

Отвечает за orchestration — запуск нескольких компонентов приложения в необходимой последовательности.

```text
jobs/
├── collect.py
├── rank.py
└── notify.py
```

## `collect.py`

Управляет сбором вакансий.

```text
Collector
    ↓
Processing
    ↓
Storage
```

## `rank.py`

Запускает pipeline ранжирования.

```text
Storage
   ↓
NLP
   ↓
Scoring
   ↓
ML
   ↓
Ranking
```

## `notify.py`

Передаёт результаты пользователю.

```text
Ranked Vacancies
       ↓
Telegram
```

`jobs` должны управлять процессами, но не содержать основную реализацию алгоритмов.

---

# 18. `tests/`

```text
tests/
```

Содержит автоматические тесты.

Структура желательно повторяет структуру `src/`.

```text
tests/
├── collectors/
├── storage/
├── processing/
├── nlp/
├── scoring/
├── ml/
├── bot/
└── jobs/
```

Пример:

```text
src/
└── processing/
    └── normalizer.py

tests/
└── processing/
    └── test_normalizer.py
```

Это упрощает поиск тестов для конкретного модуля.

---

# 19. `notebooks/`

```text
notebooks/
```

Каталог исследовательских Jupyter notebooks.

```text
notebooks/
├── 01_data_quality.ipynb
├── 02_market_eda.ipynb
├── 03_skills.ipynb
├── 04_matching.ipynb
└── 05_ml_model.ipynb
```

Назначение:

```text
EDA
анализ данных
визуализация
исследование признаков
проверка гипотез
сравнение моделей
ML-эксперименты
```

Notebooks являются исследовательской средой.

Production-код не должен существовать только внутри `.ipynb`.

После проверки алгоритм переносится в:

```text
src/
```

Этот принцип уже закреплён в проекте.

---

# 20. `data/`

```text
data/
```

Содержит локальные наборы данных.

Предлагаемая структура:

```text
data/
├── raw/
├── processed/
└── external/
```

## `data/raw/`

Исходные данные без обработки.

Например:

```text
raw HeadHunter responses
JSON exports
historical vacancy snapshots
```

Исходные данные желательно сохранять, чтобы preprocessing можно было повторить без повторного обращения к внешнему API.

## `data/processed/`

Данные после:

```text
cleaning
normalization
deduplication
feature preparation
```

## `data/external/`

Дополнительные сторонние наборы данных.

Например:

```text
skill taxonomy
location dictionary
salary reference data
```

если они понадобятся проекту.

## Git

Сами большие датасеты обычно не хранятся в Git.

Каталог рекомендуется добавить в `.gitignore`.

При необходимости структура сохраняется через `.gitkeep`.

---

# 21. `artifacts/`

```text
artifacts/
```

Содержит результаты работы Data Science и Machine Learning.

Это **не исходный код**.

Структура:

```text
artifacts/
├── ml_models/
├── embeddings/
├── metrics/
└── reports/
```

---

# 22. `artifacts/ml_models/`

```text
artifacts/ml_models/
```

Здесь находятся сериализованные обученные ML-модели.

Например:

```text
ranking_model.joblib
logistic_regression.pkl
gradient_boosting.joblib
```

Это отличается от:

```text
src/storage/models/
```

где находятся SQLAlchemy ORM-модели.

То есть:

```text
src/storage/models/
    ↓
Python-классы таблиц PostgreSQL

artifacts/ml_models/
    ↓
файлы обученных Machine Learning моделей
```

---

# 23. `artifacts/embeddings/`

```text
artifacts/embeddings/
```

Может использоваться для локального хранения вычисленных embedding-векторов или их экспортов.

Например:

```text
vacancy_embeddings.npy
candidate_embeddings.npy
```

При переходе на `pgvector` часть embeddings может храниться непосредственно в PostgreSQL.

---

# 24. `artifacts/metrics/`

```text
artifacts/metrics/
```

Результаты оценки моделей.

Например:

```text
baseline_metrics.json
logistic_regression_metrics.json
experiment_001.json
```

В ML-экспериментах необходимо фиксировать:

```text
dataset
preprocessing
features
model
hyperparameters
metrics
train period
test period
```

---

# 25. `artifacts/reports/`

```text
artifacts/reports/
```

Автоматически генерируемые аналитические отчёты.

Например:

```text
market_analysis.html
model_comparison.html
data_quality_report.html
```

Если отчёт является частью официальной документации проекта, его итоговая версия может быть перенесена в `docs/`.

---

# 26. `alembic/`

```text
alembic/
```

Содержит миграции PostgreSQL.

```text
alembic/
├── versions/
└── env.py
```

## `versions/`

История изменений схемы базы данных.

Например:

```text
001_create_vacancies.py
002_create_candidates.py
003_add_feedback.py
```

## `env.py`

Конфигурация среды Alembic.

Изменения схемы общей базы данных должны сопровождаться migration.

---

# 27. `docs/`

```text
docs/
```

Содержит расширенную документацию проекта.

Пример:

```text
docs/
├── ARCHITECTURE.md
└── GIT_GUIDELINES.md
```

## `ARCHITECTURE.md`

Описание архитектуры, модулей и потоков данных.

## `GIT_GUIDELINES.md`

Правила:

```text
ветвления
commit
Pull Request
Code Review
разрешения конфликтов
```

В дальнейшем здесь могут появиться:

```text
DATABASE.md
API.md
ML_PIPELINE.md
DEPLOYMENT.md
```

---

# 28. `docker/`

```text
docker/
```

Содержит дополнительные Docker-конфигурации.

Например:

```text
docker/
├── postgres/
├── scripts/
└── configs/
```

Используется, если инфраструктурные файлы становятся слишком большими для размещения только в корне проекта.

---

# 29. `.env.example`

```text
.env.example
```

Шаблон переменных окружения.

Например:

```text
DATABASE_URL=
TELEGRAM_BOT_TOKEN=
HH_CLIENT_ID=
HH_CLIENT_SECRET=
```

Файл не должен содержать реальные секреты.

Сам `.env` хранится только локально и добавляется в `.gitignore`.

---

# 30. `.gitignore`

```text
.gitignore
```

Определяет файлы, которые не должны попадать в Git.

Например:

```text
.env
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/

data/
artifacts/

.ipynb_checkpoints/
```

---

# 31. `.pre-commit-config.yaml`

```text
.pre-commit-config.yaml
```

Настройка автоматических проверок перед созданием commit.

Может запускать:

```text
ruff
formatting
basic file checks
```

Проект уже предполагает использование `pre-commit` как инструмента контроля качества.

---

# 32. `alembic.ini`

```text
alembic.ini
```

Основной конфигурационный файл Alembic.

Используется командами:

```bash
alembic revision
alembic upgrade
alembic downgrade
```

---

# 33. `docker-compose.yml`

```text
docker-compose.yml
```

Описывает локальное окружение проекта.

Например:

```text
application
PostgreSQL
```

В дальнейшем здесь могут появиться дополнительные инфраструктурные компоненты.

---

# 34. `Dockerfile`

```text
Dockerfile
```

Описывает сборку Docker image приложения.

Используется для создания воспроизводимого окружения запуска Python-кода проекта.

---

# 35. `pyproject.toml`

```text
pyproject.toml
```

Центральный конфигурационный файл Python-проекта.

Может содержать:

```text
project metadata
dependencies
pytest configuration
ruff configuration
mypy configuration
build configuration
```

---

# 36. `README.md`

```text
README.md
```

Главная точка входа в документацию репозитория.

README должен содержать:

* краткое описание проекта;
* основные возможности;
* стек;
* быстрый старт;
* краткую схему архитектуры;
* ссылки на подробную документацию.

Например:

```text
docs/ARCHITECTURE.md
docs/GIT_GUIDELINES.md
```

---

# 37. Поток данных

Основной pipeline:

```text
External API
     ↓
Collectors
     ↓
Domain Vacancy
     ↓
Processing
     ↓
PostgreSQL
     ↓
NLP
     ↓
Scoring
     ↓
ML Ranking
     ↓
Telegram
     ↓
User Feedback
     ↓
PostgreSQL
     ↓
Training Dataset
     ↓
Model Retraining
```

Этот поток соответствует общей долгосрочной цели проекта: собирать, нормализовать, анализировать, сопоставлять, ранжировать вакансии и постепенно обучаться на feedback пользователя.

---

# 38. Направление зависимостей

Основные зависимости должны двигаться преимущественно по направлению pipeline:

```text
collectors
    ↓
domain
    ↓
processing
    ↓
storage
    ↓
nlp
    ↓
scoring
    ↓
ml
```

При этом `jobs` управляет запуском этих компонентов:

```text
        jobs
         │
 ┌───────┼────────┐
 ▼       ▼        ▼
collect rank    notify
```

`bot` является внешним пользовательским интерфейсом.

Необходимо избегать зависимостей вроде:

```text
domain ───────► Telegram

scoring ──────► HeadHunter API

ml ───────────► Telegram handlers
```

---

# 39. Основные архитектурные правила

## 39.1. Один модуль — одна ответственность

Collector получает данные.

Processing обрабатывает данные.

Storage сохраняет данные.

NLP анализирует текст.

Scoring рассчитывает baseline.

ML выполняет машинное ранжирование.

Bot взаимодействует с пользователем.

Jobs управляет процессами.

---

## 39.2. Внешние источники изолированы

HeadHunter, Habr Career и другие источники должны находиться только на уровне collectors.

Бизнес-логика не должна зависеть от конкретной структуры job-сайта.

---

## 39.3. Raw data сохраняется

Исходные данные должны сохраняться до preprocessing.

Это позволяет повторно обработать исторические данные при изменении логики обработки.

---

## 39.4. Research и Production разделяются

```text
notebooks/
```

используется для экспериментов.

```text
src/
```

используется для production-кода.

---

## 39.5. ML code и ML artifacts разделяются

```text
src/ml/
```

содержит алгоритмы и код.

```text
artifacts/ml_models/
```

содержит обученные модели.

Это предотвращает смешивание исходного кода и результатов обучения.

---

## 39.6. ORM models и ML models — разные сущности

```text
src/storage/models/
```

означает:

```text
SQLAlchemy ORM models
```

а:

```text
artifacts/ml_models/
```

означает:

```text
trained Machine Learning models
```

Эти каталоги имеют разное назначение и не дублируют друг друга.

---

# 40. Краткая карта каталогов

| Каталог                     | Назначение                           |
| --------------------------- | ------------------------------------ |
| `src/`                      | Production-код                       |
| `src/collectors/`           | Получение вакансий                   |
| `src/domain/`               | Доменные сущности                    |
| `src/storage/`              | Работа с БД                          |
| `src/storage/models/`       | SQLAlchemy ORM-модели                |
| `src/storage/repositories/` | Доступ к данным                      |
| `src/processing/`           | Очистка и нормализация               |
| `src/nlp/`                  | NLP и embeddings                     |
| `src/scoring/`              | Baseline scoring                     |
| `src/ml/`                   | Код Machine Learning                 |
| `src/letters/`              | Сопроводительные письма              |
| `src/bot/`                  | Telegram UI                          |
| `src/jobs/`                 | Orchestration                        |
| `tests/`                    | Автоматические тесты                 |
| `notebooks/`                | Исследования и EDA                   |
| `data/raw/`                 | Исходные данные                      |
| `data/processed/`           | Обработанные данные                  |
| `data/external/`            | Внешние datasets                     |
| `artifacts/ml_models/`      | Обученные ML-модели                  |
| `artifacts/embeddings/`     | Embedding-векторы                    |
| `artifacts/metrics/`        | Метрики экспериментов                |
| `artifacts/reports/`        | Сгенерированные отчёты               |
| `alembic/`                  | Миграции PostgreSQL                  |
| `docs/`                     | Документация                         |
| `docker/`                   | Дополнительная Docker-инфраструктура |

---

# 41. Итоговая схема

```text
                    Job Market Analyzer

External APIs
     │
     ▼
┌─────────────┐
│ Collectors  │
└──────┬──────┘
       ▼
┌─────────────┐
│   Domain    │
└──────┬──────┘
       ▼
┌─────────────┐
│ Processing  │
└──────┬──────┘
       ▼
┌─────────────┐
│   Storage   │
│ PostgreSQL  │
└──────┬──────┘
       │
       ├────────────► Notebooks / Analytics
       │
       ▼
┌─────────────┐
│     NLP     │
└──────┬──────┘
       ▼
┌─────────────┐
│   Scoring   │
└──────┬──────┘
       ▼
┌─────────────┐
│     ML      │──────► artifacts/
└──────┬──────┘
       ▼
┌─────────────┐
│   Ranking   │
└──────┬──────┘
       ▼
┌─────────────┐
│  Telegram   │
└──────┬──────┘
       ▼
    Feedback
       │
       └─────────────► PostgreSQL
                         │
                         ▼
                    Training Data
                         │
                         ▼
                    Model Retraining
```

Основной принцип архитектуры:

> Код, данные, инфраструктура и результаты ML должны быть разделены по назначению. Каждый каталог отвечает только за свою область системы, а взаимодействие между компонентами происходит через чётко определённые границы.
