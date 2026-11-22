import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx
import requests
import yaml
from aiogram.exceptions import TelegramAPIError
from aiogram.utils.token import TokenValidationError
from pydantic import ValidationError

from bot.delivery import TelegramSender
from collectors.base import SearchQuery
from collectors.errors import CollectionStopped, VacancyParseError
from collectors.registry import SourceAdapterError
from domain import CandidateProfile, Vacancy
from domain.matching import MatchResult
from domain.serving_ranker import RankerPolicy
from domain.vacancy_analysis import VacancyCategory
from jobs.collection_budget import AvitoRateLimited
from jobs.collection_runtime import CollectionAlreadyRunning, run_collection
from jobs.import_html import import_html_snapshots
from jobs.recommendations import recommend
from jobs.settings import Settings
from jobs.skill_gap import analyze_skill_gap
from nlp.profile_experience import sync_profile_experience
from observability import emit_event, logging_session, operation
from observability.settings import LoggingOptions
from scoring.presentation import evidence_label, score_label
from scoring.skill_gap import render_skill_gap
from storage.database import DATABASE_ERRORS, create_engine, session_factory
from storage.repositories import VacancyRepository


def load_profile(path: Path) -> CandidateProfile:
    with path.open(encoding="utf-8") as stream:
        return sync_profile_experience(CandidateProfile.model_validate(yaml.safe_load(stream)))


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return number


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's original message contains rejected values and paths.
        try:
            options = LoggingOptions()
        except (OSError, ValueError):
            options = LoggingOptions(_env_file=None, level="INFO", console_format="text")
        with (
            logging_session(level=options.level, console_format=options.console_format),
            operation("command"),
        ):
            emit_event(
                logging.getLogger("jobs.cli"),
                "application.failed",
                level=logging.ERROR,
                code="invalid_arguments",
                result="failed",
            )
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = SafeArgumentParser(
        prog="vacancy-analyzer", description="Сбор и объяснимое ранжирование вакансий"
    )
    commands = root.add_subparsers(dest="command", required=True)
    from jobs.evaluation_cli import add_commands

    add_commands(commands)
    resume = commands.add_parser("resume", help="Загрузить, проверить и применить резюме")
    resume.add_argument("--user-id", type=_positive, required=True)
    resume.add_argument("--file", type=Path)
    resume.add_argument("--details", action="store_true")
    resume.add_argument("--edit", nargs=2, metavar=("FIELD", "VALUE"))
    action = resume.add_mutually_exclusive_group()
    action.add_argument(
        "--apply", action="store_true", help="Показать результат и запросить подтверждение"
    )
    action.add_argument("--cancel", action="store_true")
    action.add_argument("--delete", action="store_true")
    vacancy_backfill = commands.add_parser(
        "vacancy-backfill", help="Факты/проекции каталога, актуальные сначала"
    )
    vacancy_backfill.add_argument("--batch-size", type=_positive, default=100)
    vacancy_backfill.add_argument("--max-batches", type=_positive, default=1)
    vacancy_backfill.add_argument("--restart", action="store_true")
    vacancy_backfill.add_argument("--enqueue-analysis", action="store_true")
    backfill = commands.add_parser("facts-backfill", help="Ограниченный backfill фактов без LLM")
    backfill.add_argument("--batch-size", type=_positive, default=100)
    backfill.add_argument("--max-batches", type=_positive, default=1)
    backfill.add_argument(
        "--restart", action="store_true", help="Начать повторную проверку с начала"
    )
    telegram_id = commands.add_parser("telegram-id", help="Узнать ID своего личного чата с ботом")
    telegram_id.add_argument(
        "--wait",
        type=int,
        choices=range(0, 51),
        default=30,
        metavar="0..50",
        help="Ожидание нового сообщения, по умолчанию 30 с",
    )
    collect = commands.add_parser("collect", help="Собрать IT-вакансии в общий каталог")
    mode = collect.add_mutually_exclusive_group()
    mode.add_argument(
        "--query", action="append", help="Ручной запрос; можно повторить несколько раз"
    )
    mode.add_argument(
        "--from-profiles", action="store_true", help="По направлениям из профилей пользователей"
    )
    mode.add_argument(
        "--it", action="store_true", help="IT-каталог по рубрикам площадок (по умолчанию)"
    )
    collect.add_argument("--area")
    collect.add_argument(
        "--search-field",
        choices=["all", "name"],
        default="all",
        help="Область ручного поиска; профили ищутся по всем полям, включая описание",
    )
    collect.add_argument(
        "--days",
        type=_positive,
        default=None,
        help="Ограничить период поиска HH (1–30 дней); по умолчанию без фильтра",
    )
    collect.add_argument("--pages", type=_positive, default=1)
    collect.add_argument("--per-page", type=_positive, default=20)
    collect.add_argument(
        "--new-only", action="store_true", help="Сбор новых ID с отключённой актуализацией каталога"
    )
    collect.add_argument(
        "--source",
        choices=["all", "hh", "html", "api", "habr_career", "trudvsem", "superjob", "avito"],
        default=None,
    )
    collect.add_argument(
        "--resume-source",
        action="store_true",
        help="Снять сохранённый stop после устранения 403/captcha; бюджет не сбрасывается",
    )
    collect.add_argument(
        "--watch", action="store_true", help="Продолжать обход в постоянном режиме"
    )
    commands.add_parser(
        "catalog-status", help="JSON покрытия, давности и бюджета без запросов к HH"
    )
    importer = commands.add_parser("import-html", help="Импортировать сохранённые HTML без сети")
    importer.add_argument("--source", choices=("hh", "avito"), default="hh")
    pages = importer.add_mutually_exclusive_group(required=True)
    pages.add_argument("--directory", type=Path)
    pages.add_argument("--file", type=Path, help="Одна сохранённая страница Авито, HTML или MHTML")
    rank = commands.add_parser("rank", help="Ранжировать сохранённые вакансии")
    rank.add_argument("--profile", type=Path, required=True)
    rank.add_argument(
        "--days",
        type=_positive,
        default=None,
        help="Ограничить возраст вакансий; по умолчанию без ограничения",
    )
    rank.add_argument("--category", choices=[value.value for value in VacancyCategory])
    rank.add_argument(
        "--limit",
        type=_positive,
        help="Число кандидатов после поиска и обязательных проверок (по умолчанию RETRIEVAL_K)",
    )
    rank.add_argument("--top", type=_positive, default=5)
    rank.add_argument("--json", action="store_true")
    rank.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Показать также активные вакансии, не соответствующие обязательным условиям",
    )
    analyze = commands.add_parser(
        "analyze-vacancies", help="Разобрать активные вакансии через Ollama"
    )
    retrieval = commands.add_parser(
        "retrieval-evaluate", help="Проверить отбор на замороженном наборе без БД и LLM"
    )
    retrieval.add_argument("--bundle", type=Path, required=True)
    retrieval.add_argument("--output", type=Path, required=True)
    semantic = commands.add_parser(
        "semantic-audit", help="Проверить потери отбора перед смысловым поиском, без БД и моделей"
    )
    semantic.add_argument("--bundle", type=Path, required=True)
    semantic.add_argument("--output", type=Path, required=True)
    semantic.add_argument("--k", type=_positive, default=64)
    semantic.add_argument("--min-score", type=float, default=50)
    semantic_model = commands.add_parser(
        "semantic-download", help="Явно загрузить лёгкую модель смыслового поиска"
    )
    semantic_model.add_argument("--directory", type=Path, required=True)
    semantic_run = commands.add_parser(
        "semantic-experiment", help="Локальное сравнение вариантов поиска без изменения бота"
    )
    semantic_run.add_argument("--bundle", type=Path, required=True)
    semantic_run.add_argument("--review-directory", type=Path, required=True)
    semantic_run.add_argument("--model-directory", type=Path, required=True)
    semantic_run.add_argument("--output", type=Path, required=True)
    semantic_run.add_argument(
        "--reuse-index-from",
        type=Path,
        help="Переиспользовать проверенные индексы предыдущего эксперимента",
    )
    semantic_check = commands.add_parser(
        "semantic-check", help="Проверить допуск смыслового поиска по отчёту"
    )
    semantic_check.add_argument("--report", type=Path, required=True)
    semantic_check.add_argument("--channel", choices=["rrf", "novelty"], default="rrf")
    serving_train = commands.add_parser(
        "ranker-train", help="Обучить и проверить общий ранжировщик без включения в бота"
    )
    serving_train.add_argument("--bundle", type=Path, required=True)
    serving_train.add_argument("--output", type=Path, required=True)
    serving_train.add_argument("--ledger", type=Path)
    serving_train.add_argument("--draft-labels", type=Path)
    serving_train.add_argument(
        "--allow-assistant-draft",
        action="store_true",
        help="Только эксперимент; такая модель не проходит включение",
    )
    comparison = commands.add_parser(
        "ranker-compare", help="Сравнить порядки через режим shadow без БД и Telegram"
    )
    comparison.add_argument("--bundle", type=Path, required=True)
    comparison.add_argument("--model", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("ranker-check", help="Пересчитать gates по данным без final test")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--ledger", type=Path)
    check.add_argument("--draft-labels", type=Path)
    check.add_argument("--allow-assistant-draft", action="store_true")
    feedback_training = commands.add_parser(
        "feedback-training", help="Подготовить события одного пользователя для ручной проверки"
    )
    feedback_training.add_argument("--user-id", type=int, required=True)
    feedback_training.add_argument("--limit", type=_positive, default=500)
    from jobs.learning_cli import add_commands as add_learning_commands

    add_learning_commands(commands)
    analyze.add_argument("--limit", type=_positive, default=20)
    analyze.add_argument("--force", action="store_true", help="Повторить даже сохранённый разбор")
    gap = commands.add_parser("skill-gap", help="Требования вакансий, не отражённые в профиле")
    gap.add_argument("--profile", type=Path, required=True)
    gap.add_argument("--days", type=_positive, default=30)
    gap.add_argument("--limit", type=_positive, default=2000)
    gap.add_argument("--json", action="store_true")
    dataset = commands.add_parser(
        "matching-dataset", help="Подготовить текстовые признаки и пустую ручную разметку"
    )
    dataset.add_argument("--profile", type=Path, required=True)
    dataset.add_argument("--resume", type=Path, help="Локальное резюме PDF, DOCX или TXT")
    dataset.add_argument("--output", type=Path, required=True, help="Новый каталог эксперимента")
    dataset.add_argument("--days", type=_positive, default=30)
    dataset.add_argument("--limit", type=_positive, default=500)
    evaluate = commands.add_parser(
        "matching-evaluate", help="Сравнить признаки с baseline по ручным оценкам, без БД"
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--top", type=_positive, default=5)
    evaluate.add_argument("--model", type=Path, help="Обученный локальный model.json")
    label = commands.add_parser("matching-label", help="Оценивать вакансии по одной с сохранением")
    label.add_argument("--dataset", type=Path, required=True)
    label.add_argument("--labels", type=Path, required=True)
    label.add_argument("--limit", type=_positive, default=20, help="Карточек за этот сеанс")
    train = commands.add_parser(
        "matching-train", help="Обучить свой ранжировщик с отдельной проверкой по резюме"
    )
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True, help="Новый каталог результата")
    train.add_argument("--top", type=_positive, default=5)
    benchmark = commands.add_parser(
        "benchmark-prepare", help="Зафиксировать каталог HH и профили для проверки качества"
    )
    benchmark.add_argument("--html-directory", type=Path, required=True)
    benchmark.add_argument("--cases", type=Path, default=Path("config/evaluation/benchmark.yaml"))
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--at", required=True, help="Момент оценки ISO 8601 с часовым поясом")
    benchmark.add_argument("--limit", type=_positive, default=200)
    benchmark_eval = commands.add_parser(
        "benchmark-evaluate", help="Сравнить правила с зафиксированной базой, без БД и LLM"
    )
    benchmark_eval.add_argument("--bundle", type=Path, required=True)
    benchmark_eval.add_argument("--output", type=Path, required=True, help="Новый JSON-отчёт")
    benchmark_eval.add_argument("--repeats", type=_positive, default=1)
    benchmark_label = commands.add_parser(
        "benchmark-label", help="Подтвердить оценки пар человеком с сохранением прогресса"
    )
    benchmark_label.add_argument("--bundle", type=Path, required=True)
    benchmark_label.add_argument("--case", required=True)
    benchmark_label.add_argument("--reviewer", required=True)
    benchmark_label.add_argument("--limit", type=_positive, default=20)
    commands.add_parser("bot", help="Запустить Telegram-бота для пользователей")
    worker = commands.add_parser(
        "worker", help="Исполнять сохранённые задания без Telegram и сбора HH"
    )
    worker.add_argument("--once", action="store_true", help="Выполнить не более одного задания")
    commands.add_parser(
        "inference-status", help="Очередь, модель, память и ожидание восстановления (JSON)"
    )
    notify = commands.add_parser(
        "notify", help="Отправить рекомендации зарегистрированному пользователю"
    )
    notify.add_argument("--user-id", type=_positive, required=True)
    notify.add_argument("--dry-run", action="store_true", help="Просмотреть без отправки")
    return root


def render(results: list[tuple[Vacancy, MatchResult]], *, as_json: bool) -> str:
    if as_json:
        return json.dumps(
            [
                {
                    "vacancy": vacancy.model_dump(mode="json", exclude={"raw_payload"}),
                    "match": match.model_dump(mode="json"),
                }
                for vacancy, match in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    if not results:
        return "Подходящих вакансий пока нет. Сначала выполните collect или измените профиль."
    cards = []
    for vacancy, match in results:
        lines = [
            f"{score_label(match)} — {vacancy.title}",
            vacancy.company,
            str(vacancy.url),
            evidence_label(match),
        ]
        if not match.eligible:
            lines.insert(0, "Не соответствует обязательным условиям профиля")
        lines += [f"  + {reason}" for reason in match.reasons]
        lines += [f"  ! {warning}" for warning in match.warnings]
        cards.append("\n".join(lines))
    return "\n\n".join(cards)


async def run_telegram(args: argparse.Namespace) -> None:
    from bot.runtime import create_bot, run_bot
    from jobs.users import UserService

    settings = Settings()
    if args.command == "telegram-id":
        from bot.onboarding import show_private_chat_ids

        bot = create_bot(settings)
        try:
            await show_private_chat_ids(bot, wait_seconds=args.wait)
        finally:
            await bot.session.close()
        return
    if args.command == "bot":
        await run_bot(settings)
        return
    # Validate token before database/network work, except during a dry run.
    bot = None if args.dry_run else create_bot(settings)
    engine = None
    try:
        engine = create_engine(settings.database_url)
        users = UserService(
            session_factory(engine),
            top=settings.telegram_top,
            min_score=settings.telegram_min_score,
            ranker_policy=RankerPolicy.from_settings(settings),
            saved_selections_enabled=settings.saved_selections_enabled,
        )
        service = await users.notifications(args.user_id, wait_for_rating=not args.dry_run)
        if args.dry_run:
            print(render(await service.pending(), as_json=False))
        else:
            report = await service.deliver(TelegramSender(bot))
            if report.empty_message:
                print(report.empty_message)
            print(
                f"Отправлено: {report.sent}; ошибок: {report.failed}; "
                f"без подтверждения: {report.uncertain}."
            )
            if report.failed or report.uncertain:
                raise SystemExit(1)
    finally:
        try:
            if bot is not None:
                await bot.session.close()
        finally:
            if engine is not None:
                await engine.dispose()


async def run_command(args: argparse.Namespace) -> None:
    if args.command.startswith("evaluation-"):
        from jobs.evaluation_cli import run_command as run_evaluation

        await run_evaluation(args)
        return
    if args.command == "semantic-check":
        from jobs.semantic_acceptance import acceptance

        with args.report.open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Отчёт превышает допустимый размер")
        decision = acceptance(json.loads(data), channel=args.channel)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        if not decision["allowed"]:
            raise SystemExit(2)
        return
    if args.command == "semantic-download":
        from jobs.semantic_model import download_model

        await asyncio.to_thread(download_model, args.directory)
        print("Локальные файлы модели проверены. Модель не запущена.")
        return
    if args.command == "semantic-experiment":
        from jobs.semantic_experiment import run_semantic_experiment

        result = await run_semantic_experiment(
            args.bundle,
            args.review_directory,
            args.model_directory,
            args.output,
            reuse_from=args.reuse_index_from,
        )
        print(f"Сравнено пар: {len(result['diagnostic_pairs'])}. Модель в боте не включена.")
        return
    if args.command == "semantic-audit":
        from jobs.semantic_audit import audit_semantic_need, write_semantic_audit

        if args.output == args.output.with_suffix(".md") or any(
            p.exists() for p in (args.output, args.output.with_suffix(".md"))
        ):
            raise ValueError("Выберите новые JSON и Markdown файлы отчёта")
        report = await audit_semantic_need(args.bundle, k=args.k, min_score=args.min_score)
        write_semantic_audit(report, args.output)
        print(
            f"Смысловой поиск: {report['decision']}; потерь отбора: "
            f"{len(report['retrieval_misses_for_review'])}. Выдача не изменена."
        )
        return
    if args.command == "ranker-compare":
        from jobs.ranker_comparison import compare_ranker

        if args.output.exists():
            raise ValueError("Выберите новый файл сравнения")
        report = await compare_ranker(args.bundle, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Сравнено validation-профилей: {len(report['cases'])}; выдача не изменена.")
        return
    if args.command == "ranker-check":
        from jobs.evaluation_data import write_new
        from jobs.ranker_pipeline import check

        if args.output.exists():
            raise ValueError("Choose a new immutable report")
        report = await check(
            args.bundle,
            args.model,
            ledger=args.ledger,
            draft_labels=args.draft_labels,
            allow_assistant_draft=args.allow_assistant_draft,
        )
        write_new(args.output, report)
        print(
            json.dumps(
                {
                    "model_id": report["model_id"],
                    "activation_blockers": report["activation_blockers"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.command == "ranker-train":
        from jobs.serving_ranker_training import save_serving_ranker, train_serving_ranker

        if args.output.exists():
            raise ValueError("Выберите новый каталог модели")
        artifact, report = await train_serving_ranker(
            args.bundle,
            allow_assistant_draft=args.allow_assistant_draft,
            ledger=args.ledger,
            draft_labels=args.draft_labels,
        )
        save_serving_ranker(artifact, report, args.output)
        print(
            f"Артефакт: {args.output}. Модель не включена; "
            f"причин запрета: {len(report['activation_blockers'])}."
        )
        return
    if args.command == "retrieval-evaluate":
        from jobs.retrieval_evaluation import evaluate_retrieval, write_report

        if args.output.exists() or args.output.with_suffix(".md").exists():
            raise ValueError("Выберите новый файл отчёта")
        report = await evaluate_retrieval(args.bundle)
        write_report(report, args.output)
        print(f"Отчёт отбора: {args.output}")
        return
    if args.command == "benchmark-prepare":
        from datetime import datetime

        from jobs.benchmark import prepare_benchmark

        manifest = prepare_benchmark(
            args.html_directory,
            args.cases,
            args.output,
            now=datetime.fromisoformat(args.at),
            limit=args.limit,
        )
        print(f"Подготовлено пар: {manifest['pair_count']}. Каталог: {args.output}")
        return
    if args.command == "benchmark-evaluate":
        from jobs.benchmark import evaluate_benchmark, render_benchmark_report

        markdown = args.output.with_suffix(".md")
        if args.output.exists() or markdown.exists() or markdown == args.output:
            raise ValueError("Отчёт уже существует; выберите новый файл")
        report = evaluate_benchmark(args.bundle, repeats=args.repeats)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        with markdown.open("x", encoding="utf-8") as stream:
            stream.write(render_benchmark_report(report))
        print(f"Отчёт: {args.output}; подтверждена разметка: {report['quality_ready']}")
        return
    if args.command == "benchmark-label":
        from jobs.benchmark_labels import label_benchmark

        count = label_benchmark(
            args.bundle,
            case_id=args.case,
            reviewer=args.reviewer,
            limit=args.limit,
        )
        print(f"Подтверждено оценок профиля: {count}")
        return
    if args.command == "matching-label":
        from domain.matching_dataset import MatchingDataset
        from jobs.matching_labels import label_matching_dataset

        dataset = MatchingDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
        label_matching_dataset(dataset, args.labels, limit=args.limit)
        return
    if args.command == "matching-train":
        from jobs.ranker_training import train_from_manifest, write_training_result

        if args.output.exists():
            raise ValueError("Каталог модели уже существует. Выберите новый каталог.")
        model, report = train_from_manifest(args.manifest, top=args.top)
        write_training_result(model, report, args.output)
        print(f"Модель: {args.output / 'model.json'}; проверка: {args.output / 'validation.json'}.")
        print("Модель сохранена локально. Пользовательская выдача автоматически не изменяется.")
        return
    if args.command == "matching-evaluate":
        from domain.matching_dataset import MatchingDataset
        from scoring.matching_evaluation import evaluate_matching, load_judgments

        dataset = MatchingDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
        labels = load_judgments(args.labels, dataset)
        learned = None
        independent = None
        if args.model:
            from domain.ranker import LinearRankerModel
            from scoring.linear_ranker import profile_id, ranker_scores

            model = LinearRankerModel.model_validate_json(args.model.read_text(encoding="utf-8"))
            learned = ranker_scores(model, dataset)
            independent = (
                dataset.query_id not in model.training_query_ids
                and profile_id(dataset) not in model.training_profile_ids
            )
        report = evaluate_matching(dataset, labels, top=args.top, learned_scores=learned)
        report["query_and_profile_absent_from_training"] = independent
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    resume_text = None
    if args.command == "matching-dataset":
        from processing.resume import MAX_RESUME_BYTES, read_resume

        if args.output.exists():
            raise ValueError("Каталог эксперимента уже существует. Выберите новый каталог.")
        if args.resume:
            with args.resume.open("rb") as stream:
                resume_text, _ = read_resume(stream.read(MAX_RESUME_BYTES + 1), args.resume.name)
    settings = Settings()
    if args.command == "collect":
        from_profiles = args.from_profiles or (
            not args.it and not args.query and settings.collection_mode == "profiles"
        )
        query = SearchQuery(
            text=args.query[0] if args.query else ("profile" if from_profiles else ""),
            category=None if args.query or from_profiles else "it",
            area=args.area,
            search_field=args.search_field,
            period=args.days,
            max_pages=args.pages,
            per_page=args.per_page,
        )
    elif args.command in {"rank", "skill-gap", "matching-dataset"}:
        profile = load_profile(args.profile)
    engine = create_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        if args.command.startswith("learning-"):
            from jobs.learning_cli import run as run_learning

            await run_learning(args, sessions)
            return
        if args.command == "inference-status":
            from jobs.inference_status import inference_status

            print(
                json.dumps(await inference_status(sessions, settings), ensure_ascii=False, indent=2)
            )
            return
        if args.command == "catalog-status":
            from jobs.catalog_metrics import catalog_status

            status = await catalog_status(sessions, check_interval=settings.hh_check_interval)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return
        if args.command == "resume":
            from jobs.resume_cli import run_resume_cli

            await run_resume_cli(args, settings, sessions)
            return
        if args.command == "facts-backfill":
            from jobs.fact_backfill import backfill_batch

            if args.batch_size > 500 or args.max_batches > 100:
                raise ValueError("Backfill ограничен 500 строками × 100 пакетами за запуск")
            for index in range(args.max_batches):
                result = await backfill_batch(
                    sessions, batch_size=args.batch_size, restart=args.restart and index == 0
                )
                print(json.dumps(result, ensure_ascii=False))
                if result["done"]:
                    break
            return
        if args.command == "vacancy-backfill":
            from jobs.vacancy_backfill import backfill_vacancies

            if args.batch_size > 500 or args.max_batches > 100:
                raise ValueError("Backfill ограничен 500 строками × 100 пакетами за запуск")
            for index in range(args.max_batches):
                result = await backfill_vacancies(
                    sessions,
                    batch_size=args.batch_size,
                    restart=args.restart and index == 0,
                    model=settings.ollama_model if args.enqueue_analysis else None,
                )
                print(json.dumps(result, ensure_ascii=False))
                if result["done"]:
                    break
            return
        if args.command == "feedback-training":
            from jobs.learning_privacy import export_feedback

            report = await export_feedback(sessions, args.user_id, limit=args.limit)
            print(json.dumps(report, ensure_ascii=False))
            return
        elif args.command == "worker":
            from jobs.analysis_worker import AnalysisWorker, background_analysis
            from jobs.resume_analysis import resume_analyzer
            from jobs.vacancy_analysis import vacancy_analyzer

            if args.once:
                async with (
                    vacancy_analyzer(settings) as vacancy,
                    resume_analyzer(settings) as resume,
                ):
                    outcome = await AnalysisWorker(
                        sessions,
                        vacancy_analyzer=vacancy,
                        resume_analyzer=resume,
                        lease_seconds=max(settings.ollama_timeout, settings.resume_total_timeout)
                        + 60,
                    ).run_once()
                    print(outcome or "Нет готового задания или слот занят.")
            else:
                async with background_analysis(settings, engine, enabled=True):
                    await asyncio.Event().wait()
            return
        if args.command == "collect":
            if args.resume_source:
                from jobs.collection_budget import resume_source

                if args.source == "all":
                    raise ValueError("Для --resume-source выберите одну площадку через --source")
                name = args.source if args.source not in {None, "api", "html", "hh"} else "hh"
                await resume_source(sessions, name)
            await run_collection(
                settings,
                engine,
                query,
                watch=args.watch,
                new_only=args.new_only,
                query_texts=args.query,
                source=args.source,
            )
        elif args.command == "analyze-vacancies":
            from jobs.analysis_queue import enqueue_catalog

            if not settings.vacancy_llm_enabled:
                raise ValueError("Включите VACANCY_LLM_ENABLED в локальном .env")
            count = await enqueue_catalog(
                sessions, model=settings.ollama_model, limit=args.limit, force=args.force
            )
            print(f"Заданий в очереди: {count}. Их обработает бот или команда worker.")
        elif args.command == "import-html":
            if args.source == "avito":
                from jobs.import_avito import import_avito_pages

                report = await import_avito_pages(args.file or args.directory, sessions)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                if len(report["errors"]) == report["files"]:
                    raise SystemExit(1)
                return
            if args.file is not None:
                raise ValueError("Для импорта hh укажите --directory")
            async with sessions.begin() as session:
                count = await import_html_snapshots(
                    args.directory, VacancyRepository(session), base_url=settings.hh_base_url
                )
            print(f"Импортировано HTML-снимков: {count}")
        elif args.command == "matching-dataset":
            from jobs.matching_dataset import prepare_matching_dataset, write_matching_dataset

            async with sessions() as session:
                dataset = await prepare_matching_dataset(
                    VacancyRepository(session),
                    profile,
                    days=args.days,
                    limit=args.limit,
                    resume_text=resume_text,
                )
            write_matching_dataset(dataset, args.output)
            print(
                f"Подготовлено пар: {len(dataset.pairs)}. "
                f"Признаки: {args.output / 'dataset.json'}; "
                f"ручные оценки: {args.output / 'labels.csv'}."
            )
            for warning in dataset.warnings:
                print(warning)
        elif args.command == "skill-gap":
            async with sessions() as session:
                report = await analyze_skill_gap(
                    VacancyRepository(session),
                    profile=profile,
                    days=args.days,
                    limit=args.limit,
                )
            if args.json:
                print(report.model_dump_json(indent=2))
            else:
                print(render_skill_gap(report))
        else:
            async with sessions() as session:
                repository = VacancyRepository(session)
                result = await recommend(
                    repository,
                    profile,
                    days=args.days,
                    include_ineligible=args.include_ineligible,
                    limit=args.limit,
                    category=VacancyCategory(args.category) if args.category else None,
                    ranker_policy=RankerPolicy.from_settings(settings),
                    min_score=settings.telegram_min_score,
                    top=min(args.top, 20),
                )
                results = result.ranking if args.include_ineligible else result.admitted
                print(render(results[: args.top], as_json=args.json))
    finally:
        await engine.dispose()


def _execute(args) -> None:
    logger = logging.getLogger("jobs.cli")

    def failed(code, error, **fields):
        emit_event(
            logger,
            "application.failed",
            level=logging.ERROR,
            error=error,
            code=code,
            result="failed",
            **fields,
        )

    try:
        if args.command in {"bot", "notify", "telegram-id"}:
            asyncio.run(run_telegram(args))
        else:
            asyncio.run(run_command(args))
    except SourceAdapterError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except CollectionAlreadyRunning as exc:
        failed("already_running", exc)
        raise SystemExit(1) from None
    except AvitoRateLimited as exc:
        failed(
            exc.code,
            exc,
            source_kind="avito",
            status_code=429,
            next_run_at=exc.retry_at,
        )
        raise SystemExit(1) from None
    except CollectionStopped as exc:
        failed(
            exc.code,
            exc,
            status_code=exc.status_code,
            stage=exc.stage,
            source_vacancy_id=exc.source_vacancy_id,
        )
        raise SystemExit(1) from None
    except VacancyParseError as exc:
        failed("invalid_html", exc)
        raise SystemExit(1) from None
    except requests.HTTPError as exc:
        failed(
            "http_error",
            exc,
            status_code=exc.response.status_code if exc.response is not None else None,
        )
        raise SystemExit(1) from None
    except requests.RequestException as exc:
        failed("network_error", exc)
        raise SystemExit(1) from None
    except TokenValidationError as exc:
        failed("invalid_token", exc)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        emit_event(logger, "application.cancelled", result="cancelled")
        raise SystemExit(130) from None
    except TelegramAPIError as exc:
        failed("telegram_error", exc)
        raise SystemExit(1) from None
    except httpx.HTTPStatusError as exc:
        failed("http_error", exc, status_code=exc.response.status_code)
        raise SystemExit(1) from None
    except httpx.RequestError as exc:
        failed("network_error", exc)
        raise SystemExit(1) from None
    except DATABASE_ERRORS as exc:
        failed("database_unavailable", exc)
        raise SystemExit(1) from None
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        failed("invalid_configuration", exc)
        raise SystemExit(1) from None
    except Exception as exc:
        failed("unexpected_error", exc)
        raise SystemExit(1) from None


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parser().parse_args()
    try:
        options = LoggingOptions()
    except (OSError, ValueError):
        print(
            "Неверные настройки журнала. Проверьте LOG_LEVEL, LOG_CONSOLE_FORMAT и лимиты LOG_*.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    role = args.command if args.command in {"bot", "collect", "worker"} else "cli"
    with (
        logging_session(
            role=role,
            level=options.level,
            console_format=options.console_format,
            file_options=options,
        ),
        operation("command"),
    ):
        import time

        started = time.monotonic()
        emit_event(logging.getLogger("jobs.cli"), "application.started", result="ready")
        try:
            _execute(args)
        except SystemExit as exc:
            exit_code = (
                0
                if exc.code is None
                else (exc.code if type(exc.code) is int and 0 <= exc.code <= 255 else 1)
            )
            emit_event(
                logging.getLogger("jobs.cli"),
                "application.finished",
                level=logging.INFO
                if exit_code == 0
                else (logging.WARNING if exit_code in {2, 130} else logging.ERROR),
                result="cancelled" if exit_code == 130 else ("failed" if exit_code else "success"),
                exit_code=exit_code,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise SystemExit(exit_code) from None
        else:
            emit_event(
                logging.getLogger("jobs.cli"),
                "application.finished",
                result="success",
                duration_ms=(time.monotonic() - started) * 1000,
            )


if __name__ == "__main__":
    main()
