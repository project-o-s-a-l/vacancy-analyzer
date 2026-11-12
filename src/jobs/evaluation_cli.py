"""Public preparation, review/import, measurement and comparison commands."""

import json
from pathlib import Path

from domain.evaluation import SPLITS
from jobs.evaluation_resources import evaluation_resource


def add_commands(commands):
    prepare = commands.add_parser("evaluation-prepare", help="freeze synthetic data and protocol")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path)
    prepare.add_argument(
        "--data", type=Path, help="Freeze validated v2 data instead of synthetic data"
    )
    validate = commands.add_parser(
        "evaluation-validate", help="Validate frozen data without scoring"
    )
    validate.add_argument("--bundle", type=Path, required=True)
    validate.add_argument("--allow-legacy", action="store_true", help="Archival v1 inspection only")
    migrate = commands.add_parser(
        "evaluation-migrate-units", help="Explicit v1 month -> MONTH diagnostic copy, no training"
    )
    migrate.add_argument("--bundle", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)
    migrate.add_argument("--draft-labels", type=Path, help="Optional machine-only non-test labels")
    review = commands.add_parser("evaluation-review", help="export blind human review package")
    review.add_argument("--bundle", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--split", choices=SPLITS, default="development")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--role", choices=["reviewer", "adjudicator"], default="reviewer")
    review.add_argument("--privacy-review", type=Path)
    label = commands.add_parser("evaluation-import", help="import attested human reviews")
    for key in ("bundle", "assignment", "submission", "ledger"):
        label.add_argument("--" + key, type=Path, required=True)
    label.add_argument("--attest-human", action="store_true")
    run = commands.add_parser("evaluation-run", help="evaluate the actual serving selection path")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--ledger", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=SPLITS, default="development")
    run.add_argument("--repeats", type=int)
    run.add_argument("--clients", type=int, default=1)
    run.add_argument("--k", type=int)
    run.add_argument("--min-score", type=float)
    run.add_argument("--final-test", action="store_true")
    run.add_argument("--release-id")
    run.add_argument("--test-role", choices=["reference", "candidate"])
    run.add_argument("--ranker-mode", choices=["rules", "shadow", "model"], default="rules")
    run.add_argument("--ranker-model", type=Path)
    run.add_argument(
        "--experimental-ranker",
        action="store_true",
        help="Frozen offline replay only; never authorizes live activation",
    )
    comparison = commands.add_parser("evaluation-compare", help="paired cluster comparison")
    for key in ("reference", "candidate", "output"):
        comparison.add_argument("--" + key, type=Path, required=True)
    comparison.add_argument("--claim-improvement", action="store_true")


async def run_command(args):
    from jobs.evaluation import evaluate, save_report
    from jobs.evaluation_compare import compare, read_report
    from jobs.evaluation_data import DATA_VERSION, digest, load_data, prepare, write_new
    from jobs.evaluation_labels import export_review, import_review

    if args.command == "evaluation-prepare":
        data, _, bundle_hash = prepare(
            args.output, args.protocol or evaluation_resource("protocol.json"), data_path=args.data
        )
        result = {
            "profiles": len(data.profiles),
            "vacancies": len(data.vacancies),
            "bundle_hash": bundle_hash,
        }
    elif args.command == "evaluation-validate":
        data, _, bundle_hash = load_data(args.bundle, allow_legacy=args.allow_legacy)
        result = {
            "bundle_hash": bundle_hash,
            "data_version": data.version,
            "unit_contract_validated": data.version == DATA_VERSION,
            "status": "validated" if data.version == DATA_VERSION else "legacy_archive_only",
            "quality": "waiting_external",
            "profiles": len(data.profiles),
            "vacancies": len(data.vacancies),
        }
    elif args.command == "evaluation-migrate-units":
        from jobs.evaluation_migration import migrate_units

        receipt = migrate_units(args.bundle, args.output, draft_labels=args.draft_labels)
        result = {
            key: receipt[key]
            for key in (
                "source_bundle_hash",
                "bundle_hash",
                "transferred_draft_pairs",
                "independent_holdout",
            )
        }
    elif args.command == "evaluation-review":
        result = export_review(
            args.bundle,
            args.output,
            split=args.split,
            reviewer=args.reviewer,
            role=args.role,
            privacy_review=json.loads(args.privacy_review.read_text(encoding="utf-8"))
            if args.privacy_review
            else None,
        )
    elif args.command == "evaluation-import":
        result = import_review(
            args.bundle,
            args.assignment,
            args.submission,
            args.ledger,
            attest_human=args.attest_human,
        )
    elif args.command == "evaluation-run":
        from domain.serving_ranker import RankerPolicy

        if args.output.exists():
            raise ValueError("Choose a new immutable run directory")
        # Invalid input must not consume a one-time final-test release slot.
        load_data(args.bundle)
        if args.split == "test":
            if not args.final_test or not args.release_id or not args.test_role:
                raise ValueError("Final test requires --final-test --release-id --test-role")
            marker = (
                args.bundle
                / "test-releases"
                / (digest([args.release_id, args.test_role]) + ".json")
            )
            write_new(
                marker,
                {
                    "release_id": args.release_id,
                    "test_role": args.test_role,
                    "output": str(args.output),
                },
            )
        report = await evaluate(
            args.bundle,
            split=args.split,
            ledger=args.ledger,
            repeats=args.repeats,
            clients=args.clients,
            k=args.k,
            min_score=args.min_score,
            final_test=args.final_test,
            release_id=args.release_id,
            ranker_policy=RankerPolicy(mode=args.ranker_mode, model_path=args.ranker_model),
            experimental_ranker=args.experimental_ranker,
        )
        report["test_role"] = args.test_role
        save_report(report, args.output)
        result = {
            "quality": report["quality"],
            "result_hash": report["deterministic_result_sha256"],
        }
    else:
        result = compare(
            read_report(args.reference),
            read_report(args.candidate),
            claim_improvement=args.claim_improvement,
        )
        write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
