# Job Market Analyzer — Development Guide

Этот документ описывает единый локальный процесс разработки для участников проекта:
подготовку окружения, проверку изменений, создание коммитов и подготовку Pull Request.

Правила ветвления, именования веток и оформления Pull Request описаны отдельно в
[Git Workflow](<Job Market Analyz GitWorkflow .md>).

## 1. Первоначальная настройка

Проект поддерживает Python 3.12 и новее.

Создать виртуальное окружение в PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --group dev
```

Проверить установленные инструменты:

```powershell
python --version
ruff --version
pytest --version
```

Каталог `.venv/` является локальным и не добавляется в репозиторий.

## 2. Начало работы над задачей

Каждая задача выполняется в отдельной ветке, созданной от актуального `develop`.

Формат имени feature-ветки:

```text
feature/<module>/<short-description>
```

Пример:

```text
feature/collectors/hh-integration
```

Одна ветка и один Pull Request должны решать одну логическую задачу.

## 3. Проверка завершённой работы

После завершения реализации активировать окружение:

```powershell
.venv\Scripts\Activate.ps1
```

Запустить обязательные локальные проверки:

```powershell
ruff check .
ruff format --check .
pytest
```

Автоматический hook перед коммитом не используется. Ответственность за запуск всех трёх
проверок лежит на авторе изменений.

Если Ruff обнаружил автоматически исправимые проблемы:

```powershell
ruff check --fix .
ruff format .
```

После исправлений полный набор проверок необходимо запустить повторно.

## 4. Просмотр изменений

Перед подготовкой коммита проверить рабочую область:

```powershell
git status
git diff
```

Необходимо убедиться, что:

- изменения относятся к текущей задаче;
- отсутствуют токены, пароли и другие секреты;
- отсутствуют локальные данные и ML-артефакты;
- отсутствуют временные файлы и настройки IDE;
- новая функциональность сопровождается тестами, если это применимо;
- документация обновлена при изменении публичного поведения.

## 5. Подготовка коммита

Добавлять рекомендуется только конкретные файлы:

```powershell
git add src/collectors/hh.py
git add tests/collectors/test_hh.py
```

Для выбора отдельных изменений внутри файлов:

```powershell
git add -p
```

Без предварительной проверки не рекомендуется использовать `git add .`, поскольку в проекте
могут находиться локальные данные, notebooks и ML-артефакты.

Проверить содержимое будущего коммита:

```powershell
git status
git diff --cached
```

## 6. Формат сообщения коммита

Формат:

```text
<type>[<scope>]: <short description>
```

Примеры:

```text
feat[collectors]: add HeadHunter vacancy collector
fix[processing]: fix vacancy deduplication
test[scoring]: add baseline matching tests
docs[readme]: document local development setup
chore[infra]: configure code quality tooling
```

Коммит должен быть атомарным и содержать только один логический шаг.

## 7. Создание коммита

После успешного ручного запуска Ruff и pytest создать коммит:

```powershell
git commit -m "feat[collectors]: add HeadHunter vacancy collector"
```

Так как автоматический hook не используется, Git не повторяет проверки. Перед коммитом автор
должен убедиться, что последний запуск завершился успешно и после него файлы не изменялись.

## 8. Если проверка завершилась ошибкой

### Ruff

Исправить указанную проблему или применить безопасные автоисправления:

```powershell
ruff check --fix .
ruff format .
```

Затем повторить проверку без изменения файлов:

```powershell
ruff check .
ruff format --check .
```

### Pytest

Исправить реализацию или тест, затем повторить:

```powershell
pytest
```

Все тесты должны завершиться успешно. Предупреждения и пропущенные тесты необходимо оценивать,
а не игнорировать автоматически.

## 9. После успешного коммита

Проверить состояние рабочей области:

```powershell
git status
git log -1
```

Отправить рабочую ветку:

```powershell
git push
```

Для первой отправки новой ветки:

```powershell
git push -u origin feature/<module>/<short-description>
```

После этого создаётся Pull Request из рабочей ветки в `develop`.

Прямой push в `develop` и `main` запрещён.

## 10. Чек-лист перед Pull Request

- [ ] Pull Request решает одну логическую задачу.
- [ ] В изменениях отсутствуют секреты, данные и ML-артефакты.
- [ ] `ruff check .` проходит успешно.
- [ ] `ruff format --check .` проходит успешно.
- [ ] `pytest` проходит успешно.
- [ ] Добавлены или обновлены необходимые тесты.
- [ ] При необходимости обновлена документация.
- [ ] Просмотрен итоговый diff.

## 11. Краткий сценарий

```powershell
.venv\Scripts\Activate.ps1

git status
git diff

ruff check .
ruff format --check .
pytest

git add <files>
git diff --cached

git commit -m "<type>[<scope>]: <short description>"

git status
git push
```
