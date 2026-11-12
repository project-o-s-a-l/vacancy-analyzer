"""Local operator CLI. Database and private-root filesystem access are required."""

import json
from datetime import datetime
from pathlib import Path


def add_commands(commands):
    access = commands.add_parser("learning-access", help="Recorded consent or revoke")
    access.add_argument("--user-id", type=int, required=True)
    access.add_argument("--action", choices=["grant", "revoke", "erase"], required=True)
    access.add_argument("--consent-ref")
    commands.add_parser("learning-maintain", help="Retry revoked-file purge and expire telemetry")
    monitor = commands.add_parser(
        "learning-monitor", help="Compare proxy windows, never auto-promote"
    )
    monitor.add_argument("--user-id", type=int, required=True)
    for name in ("start", "boundary", "end"):
        monitor.add_argument("--" + name, type=datetime.fromisoformat, required=True)
    retrain = commands.add_parser(
        "learning-retrain", help="Snapshot/train/calibrate/compare/register, no activation"
    )
    for name in ("bundle", "ledger", "draft-labels", "lineage"):
        retrain.add_argument("--" + name, type=Path, required=name == "bundle")
    retrain.add_argument("--allow-assistant-draft", action="store_true")
    review = commands.add_parser("learning-review", help="Managed source review with lineage")
    for name in ("bundle", "lineage", "privacy-review"):
        review.add_argument("--" + name, type=Path, required=True)
    review.add_argument(
        "--split", choices=["train", "development", "calibration", "test"], required=True
    )
    review.add_argument("--reviewer", required=True)
    review.add_argument("--role", choices=["reviewer", "adjudicator"], default="reviewer")
    for name in ("learning-check", "learning-release"):
        command = commands.add_parser(name)
        command.add_argument("--candidate", required=True)
        if name == "learning-release":
            command.add_argument("--resource", type=Path, required=True)


async def run(args, sessions):
    from jobs import learning_privacy as privacy

    if args.command == "learning-access":
        if args.action == "grant":
            await privacy.grant(sessions, args.user_id, args.consent_ref)
            result = {"access": "granted"}
        else:
            result = await privacy.revoke(sessions, args.user_id, erase=args.action == "erase")
    elif args.command == "learning-maintain":
        from jobs.feedback_monitoring import prune

        result = {
            "purged_files": await privacy.purge_revoked(sessions),
            "expired_telemetry": await prune(sessions),
        }
    elif args.command == "learning-monitor":
        from jobs.feedback_monitoring import monitor

        result = await monitor(
            sessions, args.user_id, start=args.start, boundary=args.boundary, end=args.end
        )
    elif args.command == "learning-retrain":
        from jobs.retraining import retrain

        report = await retrain(
            sessions,
            args.bundle,
            ledger=args.ledger,
            draft_labels=args.draft_labels,
            allow_assistant_draft=args.allow_assistant_draft,
            lineage=json.loads(args.lineage.read_text(encoding="utf-8")) if args.lineage else None,
        )
        result = {
            key: report[key]
            for key in (
                "artifact_id",
                "model_id",
                "activation_blockers",
                "comparison_status",
                "activated",
            )
        }
    elif args.command == "learning-review":
        from jobs.learning_review import review

        result = await review(
            sessions,
            args.bundle,
            lineage=json.loads(args.lineage.read_text(encoding="utf-8")),
            privacy_review=json.loads(args.privacy_review.read_text(encoding="utf-8")),
            split=args.split,
            reviewer=args.reviewer,
            role=args.role,
        )
    elif args.command == "learning-check":
        from jobs.retraining import inspect_candidate

        _, manifest, checked, comparison = await inspect_candidate(sessions, args.candidate)
        result = {
            "model_id": manifest["model_id"],
            "recomputed_from_data": checked["recomputed_from_data"],
            "activation_blockers": checked["activation_blockers"],
            "comparison_status": comparison["status"],
        }
    else:
        from jobs.retraining import release_candidate

        result = await release_candidate(
            sessions, args.candidate, json.loads(args.resource.read_text(encoding="utf-8"))
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
