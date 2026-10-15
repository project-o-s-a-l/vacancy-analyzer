"""Public preparation, review/import, measurement and comparison commands."""

import json
from pathlib import Path

from domain.evaluation import SPLITS


def add_commands(commands):
    prepare = commands.add_parser("evaluation-prepare", help="freeze synthetic data and protocol")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, default=Path("config/evaluation/protocol.json"))
    review = commands.add_parser("evaluation-review", help="export blind human review package")
    review.add_argument("--bundle", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--split", choices=SPLITS, default="development")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--role", choices=["reviewer", "adjudicator"], default="reviewer")
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
    comparison = commands.add_parser("evaluation-compare", help="paired cluster comparison")
    for key in ("reference", "candidate", "output"):
        comparison.add_argument("--" + key, type=Path, required=True)
    comparison.add_argument("--claim-improvement", action="store_true")


async def run_command(args):
    from jobs.evaluation import evaluate, save_report
    from jobs.evaluation_compare import compare, read_report
    from jobs.evaluation_data import digest, prepare, write_new
    from jobs.evaluation_labels import export_review, import_review

    if args.command == "evaluation-prepare":
        data, _, bundle_hash = prepare(args.output, args.protocol)
        result = {
            "profiles": len(data.profiles),
            "vacancies": len(data.vacancies),
            "bundle_hash": bundle_hash,
        }
    elif args.command == "evaluation-review":
        result = export_review(
            args.bundle, args.output, split=args.split, reviewer=args.reviewer, role=args.role
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
        if args.output.exists():
            raise ValueError("Choose a new immutable run directory")
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
