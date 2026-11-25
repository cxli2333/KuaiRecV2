"""Combine generative shared-score/probe and MMoE/PLE task metrics."""

import argparse
import csv
import json
import os
from typing import Dict, List


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_label(metrics: Dict, fallback: str) -> str:
    return str(metrics.get("model_label") or metrics.get("model") or fallback)


def main(args) -> None:
    sources = [
        ("Generative Shared Score", args.shared_score),
        ("SFT + Frozen Probe", args.sft_probe),
        ("RL + Frozen Probe", args.rl_probe),
        ("MMoE", args.mmoe),
        ("PLE", args.ple),
    ]
    loaded = []
    for fallback, path in sources:
        if not path:
            continue
        metrics = load_json(path)
        classification = metrics.get("classification")
        if not isinstance(classification, dict):
            raise ValueError(f"No classification metrics in {path}")
        loaded.append((resolve_label(metrics, fallback), path, classification))
    if not loaded:
        raise ValueError("At least one metrics path is required")

    requested_tasks: List[str] = args.tasks or []
    if not requested_tasks:
        for _, _, classification in loaded:
            for task in classification:
                if task != "macro" and task not in requested_tasks:
                    requested_tasks.append(task)

    rows = []
    for label, path, classification in loaded:
        task_aucs = []
        row = {"model": label, "metrics_path": os.path.abspath(path)}
        for task in requested_tasks:
            values = classification.get(task, {})
            auc = values.get("auc")
            logloss = values.get("logloss")
            row[f"{task}_auc"] = auc
            row[f"{task}_logloss"] = logloss
            if auc is not None:
                task_aucs.append(float(auc))
        row["macro_auc_recomputed"] = (
            sum(task_aucs) / len(task_aucs) if task_aucs else None
        )
        row["valid_auc_tasks"] = len(task_aucs)
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "multitask_comparison.csv")
    json_path = os.path.join(args.output_dir, "multitask_comparison.json")
    fieldnames = ["model", "metrics_path"]
    for task in requested_tasks:
        fieldnames.extend([f"{task}_auc", f"{task}_logloss"])
    fieldnames.extend(["macro_auc_recomputed", "valid_auc_tasks"])
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "tasks": requested_tasks,
                "rows": rows,
                "metric_definition": (
                    "Per-task behavior classification on the shared deterministic "
                    "user-level holdout. Shared-score results use frozen target-item "
                    "likelihood without task-head training; probe results freeze the "
                    "checkpoint and train only behavior heads."
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] JSON: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multi-task AUC/LogLoss")
    parser.add_argument("--shared_score", default="")
    parser.add_argument("--sft_probe", default="")
    parser.add_argument("--rl_probe", default="")
    parser.add_argument("--mmoe", default="")
    parser.add_argument("--ple", default="")
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--output_dir", required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
