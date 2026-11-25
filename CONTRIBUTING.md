# Разработка

## Рабочий процесс

Ветка `main` содержит совместимые версии приложения и миграций. Для задачи создайте короткую ветку от актуальной `main`, например `feature/vacancy-filters` или `fix/catalog-expiry`. Делите изменения на коммиты, которые можно проверить и при необходимости отменить по отдельности.

Открывайте pull request в `main`. Опишите изменение для пользователя и выполненные проверки. Если меняется схема PostgreSQL, включите миграцию, порядок обновления и последствия отката. Проверяйте общий конвейер на нормализованных учебных вакансиях. Реализации и тесты конкретных площадок храните в отдельном пакете адаптеров.

## Окружение

На Windows с CPython 3.13 скрипт `scripts/bootstrap.ps1 -Extras all` создаёт окружение в `.local-tmp/dev-env`. В Linux с CPython 3.12 активируйте виртуальное окружение и установите инструменты разработки с теми же ограничениями версий, что и в CI.

```sh
python -m pip install --upgrade pip
python -m pip install -c requirements/linux-py312.txt --group dev -e '.[ml,semantic]'
```

## Проверки

```sh
ruff check .
ruff format --check .
python -m pytest -q -m "not postgres"
```

Для интеграционных тестов поднимите отдельный PostgreSQL. Команды ниже используют только одноразовый тестовый кластер.

```sh
docker compose --env-file .env.example -f docker-compose.test.yml up -d --wait
export TEST_DATABASE_URL=postgresql+asyncpg://va_test:disposable-test-only@127.0.0.1:55432/vacancy_analyzer_test
python scripts/check_postgres_migrations.py
python -m pytest -q -m postgres
docker compose --env-file .env.example -f docker-compose.test.yml down
```

В PowerShell задайте ту же переменную через `$env:TEST_DATABASE_URL = '…'`. Для полного прогона выполните `python -m pytest -q` с запущенным тестовым кластером. Команды для проверки pull request хранятся в [конфигурации CI](.github/workflows/check.yml).

## Данные

Токены и параметры подключения задавайте в локальном `.env`. Резюме, рабочие профили, HTML площадок, выгрузки Telegram, модели и дампы предназначены для локальных каталогов `data/` и `artifacts/`. Проверки в репозитории используют учебные записи.
