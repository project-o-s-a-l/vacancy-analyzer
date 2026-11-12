"""shadow comparison through the frozen serving evaluator."""

from domain.serving_ranker import RankerPolicy
from jobs.evaluation import evaluate


async def compare_ranker(bundle, model_path):
    report = await evaluate(
        bundle,
        split="development",
        repeats=2,
        ranker_policy=RankerPolicy(mode="shadow", model_path=model_path),
    )
    for case in report["cases"]:
        decisions = case["trace"]["ranking_decisions"]
        if any(d is None or d["fallback_reason"] or d["applied"] for d in decisions.values()):
            raise ValueError("Shadow artifact unavailable or applied unexpectedly")
        case["model_order"] = sorted(decisions, key=lambda key: decisions[key]["model_rank"])
        case["baseline_order"] = case["trace"]["ranked"]
    report["live_messages_sent"] = 0
    return report
