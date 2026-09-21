# Job Market Analyzer — Git Workflow

## 1. Основные ветки

### `main`

* Содержит стабильную и проверенную версию проекта.
* Должна соответствовать рабочему состоянию приложения.
* Merge разрешается только через Pull Request из `develop` или `hotfix/*`.
* Прямые commit и push запрещены.

### `develop`

* Основная интеграционная ветка проекта.
* Все feature-ветки вливаются сначала в `develop`.
* Merge выполняется только через Pull Request.
* После стабилизации версии `develop` вливается в `main`.
* Прямые commit и push запрещены.

---

# 2. Модули проекта

Название модуля используется в именах веток, commit и Pull Request.

| Модуль       | Назначение                                                           |
| ------------ | -------------------------------------------------------------------- |
| `collectors` | интеграции с источниками вакансий: HeadHunter, Habr Career, SuperJob |
| `domain`     | доменные модели: Vacancy, Candidate, Feedback                        |
| `storage`    | PostgreSQL, SQLAlchemy, repositories, database layer                 |
| `processing` | очистка, нормализация и дедупликация данных                          |
| `nlp`        | skill extraction, embeddings, NLP                                    |
| `scoring`    | baseline matching и расчёт score                                     |
| `ml`         | features, обучение, prediction и evaluation ML-моделей               |
| `letters`    | генерация сопроводительных писем                                     |
| `bot`        | Telegram-бот, handlers и keyboards                                   |
| `jobs`       | фоновые процессы сбора, ranking и отправки вакансий                  |
| `notebooks`  | EDA, исследования и ML-эксперименты                                  |
| `db`         | Alembic migrations и изменения структуры БД                          |
| `infra`      | Docker, Docker Compose, CI/CD и инфраструктура                       |
| `tests`      | самостоятельные задачи по тестированию                               |
| `docs`       | документация проекта                                                 |

---

# 3. Feature-ветки

## Создание

Feature-ветка создаётся от актуального `develop`.

```bash
git checkout develop
git pull
git checkout -b feature/<module>/<short-description>
```

Пример:

```bash
git checkout -b feature/collectors/hh-integration
```

## Формат имени

```text
feature/<module>/<short-description>
```

## Примеры

```text
feature/collectors/hh-integration
feature/storage/vacancy-repository
feature/processing/vacancy-normalization
feature/nlp/skill-extraction
feature/scoring/baseline-matching
feature/ml/ranking-model
feature/bot/vacancy-card
feature/jobs/vacancy-collector
feature/db/add-feedback-table
feature/docs/readme-update
```

## Правила

1. Одна ветка соответствует одной логической задаче.
2. Ветка создаётся от `develop`.
3. Merge в `develop` выполняется только через Pull Request.
4. После успешного merge ветка удаляется.
5. Продолжительность жизни feature-ветки должна быть минимальной.
6. По возможности feature-ветка не должна зависеть от другой незавершённой feature-ветки.
7. Код, необходимый для проверки функциональности, включая тесты, должен находиться в той же ветке.

---

# 4. Синхронизация с `develop`

Feature-ветку необходимо регулярно обновлять.

Перед Pull Request синхронизация обязательна.

```bash
git checkout develop
git pull

git checkout feature/<module>/<short-description>
git merge develop
```

После merge:

1. Разрешить конфликты.
2. Проверить работу приложения.
3. Запустить тесты.
4. Запустить проверки качества кода.
5. Отправить изменения.

```bash
git push
```

PR с нерешёнными конфликтами не принимается.

---

# 5. Зависимые задачи

Если одна задача зависит от другой незавершённой задачи, зависимость должна быть явно указана в Pull Request.

Пример:

```text
Depends on: feature/storage/vacancy-repository
```

Пример зависимости:

```text
feature/storage/vacancy-repository
        ↓
feature/scoring/baseline-matching
```

Предпочтительный порядок:

```text
vacancy-repository
        ↓
develop
        ↓
обновление baseline-matching
        ↓
develop
```

После merge родительской ветки:

```bash
git checkout develop
git pull

git checkout feature/scoring/baseline-matching
git merge develop
```

Создавать цепочки из большого количества зависимых feature-веток не рекомендуется.

---

# 6. Hotfix-ветки

Hotfix используется только для исправления критической ошибки в стабильной версии из `main`.

## Создание

```bash
git checkout main
git pull
git checkout -b hotfix/<short-description>
```

## Формат

```text
hotfix/<short-description>
```

## Примеры

```text
hotfix/hh-collector-crash
hotfix/database-connection
hotfix/ranking-error
```

## После исправления

Изменения должны попасть в:

```text
hotfix/*
   ├──→ main
   └──→ develop
```

Это необходимо, чтобы исправленная ошибка не появилась повторно при следующем merge `develop → main`.

Подветки от `hotfix/*` не создаются.

---

# 7. Test-ветки

Обычно тесты создаются непосредственно в feature-ветке вместе с функциональностью.

Например:

```text
feature/collectors/hh-integration
```

может содержать одновременно:

```text
src/collectors/hh.py
tests/test_hh_collector.py
```

Отдельная ветка `test/*` создаётся только тогда, когда тестирование является самостоятельной задачей.

## Формат

```text
test/<module>/<short-description>
```

## Примеры

```text
test/collectors/hh-integration
test/scoring/baseline-matching
test/storage/vacancy-repository
```

Такие ветки создаются от `develop` и вливаются обратно в `develop` через Pull Request.

---

# 8. Experiment-ветки

Для Data Science и ML допускаются экспериментальные ветки.

## Формат

```text
experiment/<module>/<short-description>
```

## Примеры

```text
experiment/ml/logistic-regression
experiment/ml/gradient-boosting
experiment/nlp/embedding-model
experiment/scoring/skill-weights
```

Экспериментальная ветка используется для:

* Jupyter notebooks;
* EDA;
* проверки гипотез;
* сравнения моделей;
* feature engineering;
* настройки scoring;
* прототипирования NLP.

Экспериментальный код не должен автоматически становиться production-кодом.

Если эксперимент признан успешным, production-реализация переносится в соответствующий модуль:

```text
src/ml/
src/nlp/
src/scoring/
```

Перед merge эксперимент должен быть воспроизводимым.

Для ML-эксперимента необходимо зафиксировать:

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

Если новая модель заменяет существующий baseline, в PR должны быть приведены результаты сравнения.

---

# 9. Коммиты

## Формат

```text
<type>[<scope>]: <short description>
```

При необходимости:

```text
<type>[<scope>]: <short description>

Detailed description.
```

## Типы

| Тип          | Назначение                                       |
| ------------ | ------------------------------------------------ |
| `feat`       | новая функциональность                           |
| `fix`        | исправление ошибки                               |
| `docs`       | документация                                     |
| `style`      | форматирование                                   |
| `refactor`   | изменение структуры кода без изменения поведения |
| `perf`       | оптимизация производительности                   |
| `test`       | тесты                                            |
| `chore`      | инфраструктура, зависимости, конфигурация        |
| `data`       | изменения подготовки или структуры данных        |
| `experiment` | исследовательские изменения                      |

## Примеры

```text
feat[collectors]: add HeadHunter vacancy collector

feat[storage]: add vacancy repository

feat[nlp]: add rule-based skill extraction

feat[scoring]: implement baseline matching

fix[processing]: fix vacancy deduplication

test[collectors]: add HH collector tests

docs[readme]: update project architecture

chore[infra]: add PostgreSQL to Docker Compose

experiment[ml]: compare ranking models
```

## Правила

1. Один логический шаг = один commit.
2. Commit должен быть атомарным.
3. Commit не должен содержать несвязанные изменения.
4. Заголовок должен быть коротким и информативным.
5. Рекомендуемая максимальная длина заголовка — 72 символа.
6. Commit message пишется на английском языке.
7. Scope должен отражать затронутый модуль.

Плохой пример:

```text
update project
```

Хороший пример:

```text
feat[collectors]: add HH vacancy search
```

---

# 10. Pull Request

## Title

```text
<module>: <short description>
```

Примеры:

```text
collectors: Add HeadHunter integration

storage: Add vacancy repository

scoring: Implement baseline matching

ml: Add initial ranking model
```

---

## Description

Рекомендуемый шаблон:

```markdown
## Description

What was implemented and why.

## Changes

- change 1
- change 2
- change 3

## Module

<module>

## Testing

Describe how the changes were tested.

## Dependencies

None
```

Если есть зависимость:

```text
Depends on: feature/storage/vacancy-repository
```

---

# 11. Pull Request для ML / Data Science

Для изменений в:

```text
ml
nlp
scoring
notebooks
```

PR при необходимости должен дополнительно содержать результаты эксперимента.

Пример:

```markdown
## Experiment

Baseline:
NDCG@5: 0.61

New model:
NDCG@5: 0.68

## Dataset

Training period:
2026-01 — 2026-07

Validation period:
2026-08

## Changes

- added new features;
- trained Logistic Regression;
- added evaluation;
- compared model with baseline.
```

Модель не должна заменять baseline только потому, что она сложнее.

Необходимо измеримое улучшение качества.

---

# 12. Требования к Pull Request

Перед merge необходимо:

1. Получить минимум один approval.
2. Разрешить все конфликты.
3. Проверить сборку проекта.
4. Запустить соответствующие тесты.
5. Проверить отсутствие случайно добавленных файлов.
6. Проверить документацию при изменении публичного поведения системы.
7. Убедиться, что PR решает одну логическую задачу.

Для Python-кода должны проходить применимые проверки:

```bash
pytest
ruff check .
mypy .
```

После merge ветка удаляется.

Описание PR пишется на английском языке.

---

# 13. Работа с notebooks

Jupyter notebooks предназначены для исследований и экспериментов.

```text
notebooks/
```

Используются для:

```text
EDA
data quality analysis
feature research
ML experiments
model comparison
visualization
```

Production-логика не должна оставаться только внутри notebook.

После подтверждения работоспособности код переносится в соответствующий модуль:

```text
src/processing/
src/nlp/
src/scoring/
src/ml/
```

Notebook должен оставаться воспроизводимым и отражать цель эксперимента.

---

# 14. Работа с базой данных

Изменение структуры PostgreSQL выполняется через Alembic migrations.

Изменения БД должны находиться в той же feature-ветке, что и связанная функциональность.

Пример:

```text
feature/storage/user-feedback
```

может содержать:

```text
SQLAlchemy model
repository
Alembic migration
tests
```

Если изменение схемы является самостоятельной задачей:

```text
feature/db/add-feedback-table
```

Не допускается изменение структуры production-БД вручную без соответствующей migration.

---

# 15. Работа с внешними источниками

Каждый новый источник вакансий реализуется внутри:

```text
src/collectors/
```

Примеры веток:

```text
feature/collectors/hh-integration
feature/collectors/habr-integration
feature/collectors/superjob-integration
```

Изменения конкретного API не должны приводить к зависимости бизнес-логики от структуры этого API.

Collector должен преобразовывать внешний формат в единую доменную модель проекта.

---

# 16. Разрешение конфликтов

Конфликты устраняет автор рабочей ветки.

## Алгоритм

Обновить целевую ветку:

```bash
git checkout develop
git pull
```

Вернуться в рабочую ветку:

```bash
git checkout feature/<module>/<short-description>
```

Влить изменения:

```bash
git merge develop
```

Исправить конфликтующие файлы.

После разрешения:

```bash
git add <file>
git commit
```

Затем:

```bash
pytest
ruff check .
```

При необходимости:

```bash
mypy .
```

После проверки:

```bash
git push
```

## Правила

* конфликты разрешаются локально;
* автор ветки отвечает за корректность результата;
* нельзя перекладывать разрешение конфликтов на reviewer;
* после разрешения конфликтов необходимо повторно проверить изменённую функциональность;
* PR с известными нерешёнными конфликтами не создаётся.

---

# 17. Процесс ветвления

Основной процесс разработки:

```text
feature/* ──────┐
test/* ─────────┤
experiment/* ───┼──→ develop ───→ main
docs feature ───┘
```

Критические исправления:

```text
main
 │
 ▼
hotfix/*
 │
 ├────────→ main
 │
 └────────→ develop
```

---

# 18. Общие требования

1. Прямой push в `main` запрещён.
2. Прямой push в `develop` запрещён.
3. Все изменения проходят через Pull Request.
4. Один PR должен решать одну логическую задачу.
5. Commit должны быть атомарными.
6. Commit и PR оформляются на английском языке.
7. Код должен соответствовать модульной структуре проекта.
8. Исследовательский код не должен смешиваться с production-кодом.
9. Новая ML-модель должна сравниваться с существующим baseline.
10. Изменения production-кода должны сопровождаться соответствующими тестами, если это применимо.
11. После merge рабочая ветка удаляется.
12. Секреты, токены, пароли и содержимое `.env` запрещено добавлять в Git.
13. Production-код размещается в `src/`.
14. Исследовательская работа выполняется в `notebooks/`.
15. Изменения схемы БД должны сопровождаться Alembic migration.

---

# 19. Краткая схема

```text
main
 ↑
 │
develop
 ↑
 ├── feature/collectors/*
 ├── feature/domain/*
 ├── feature/storage/*
 ├── feature/processing/*
 ├── feature/nlp/*
 ├── feature/scoring/*
 ├── feature/ml/*
 ├── feature/letters/*
 ├── feature/bot/*
 ├── feature/jobs/*
 ├── feature/db/*
 ├── feature/infra/*
 ├── feature/docs/*
 ├── test/*
 └── experiment/*
```

Основной принцип:

```text
задача
  ↓
ветка
  ↓
реализация + тесты
  ↓
локальная проверка
  ↓
Pull Request
  ↓
Code Review
  ↓
develop
  ↓
стабилизация
  ↓
main
```
