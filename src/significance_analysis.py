import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pytrec_eval
from beir.datasets.data_loader import GenericDataLoader
from scipy.stats import ttest_rel

from src.config.general_config import (
    DATASET_FOLDER_PATHS,
    DATASET_STRINGS,
    PLOT_OUTPUT,
    RESULTS_ROOT,
)

DATASETS = DATASET_STRINGS
METHODS = ["hyde", "hybrid", "hype"]
K = 10

METHOD_RESULT_DIRS = {
    "base": "base",
    "hyde": "hyde",
    "hybrid": "hybrid-rrf",
    "hype": "hype",
}


def find_latest_run_file(
    method: str,
    dataset: str,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    method_dir = results_root / METHOD_RESULT_DIRS[method]
    if not method_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {method_dir}")

    candidates = sorted(
        method_dir.glob(f"**/{dataset}.run.trec"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run file found for method={method}, dataset={dataset} under {method_dir}"
        )
    return candidates[0]


def _load_run(path: Path) -> dict:
    run = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, doc_id, score = parts[0], parts[2], float(parts[4])
            run.setdefault(qid, {})[doc_id] = score
    return run


def _per_query_ndcg(run: dict, qrels: dict, k: int) -> dict:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{k}"})
    scores = evaluator.evaluate(run)
    return {qid: v[f"ndcg_cut_{k}"] for qid, v in scores.items()}


def run_significance_tests(
    results_root: str | Path | None = None,
    plot_output: str | Path | None = None,
) -> list[dict]:
    del plot_output  # kept for CLI compatibility
    root = Path(results_root) if results_root else RESULTS_ROOT

    print(f"Paired t-test: each method vs. baseline, NDCG@{K}")
    print("Effect size: raw mean difference with 95% confidence interval")
    print()
    print(
        f"{'Dataset':<12} {'Method':<8} {'n':>5}  "
        f"{'d NDCG@10':>10}  {'95% CI':>20}  {'p-value':>9}"
    )
    print("-" * 72)

    results = []

    for dataset in DATASETS:
        _, _, qrels = GenericDataLoader(
            data_folder=str(DATASET_FOLDER_PATHS[dataset])
        ).load(split="test")

        base_run = _load_run(find_latest_run_file("base", dataset, root))
        base_scores = _per_query_ndcg(base_run, qrels, K)

        for method in METHODS:
            method_run = _load_run(find_latest_run_file(method, dataset, root))
            method_scores = _per_query_ndcg(method_run, qrels, K)

            qids = sorted(base_scores.keys() & method_scores.keys())
            b = [base_scores[q] for q in qids]
            m = [method_scores[q] for q in qids]
            diffs = [mi - bi for mi, bi in zip(m, b)]

            result = ttest_rel(m, b)
            ci = result.confidence_interval(confidence_level=0.95)
            p = result.pvalue
            diff = mean(diffs)

            ci_str = f"[{ci.low:+.4f}, {ci.high:+.4f}]"
            print(
                f"{dataset:<12} {method:<8} {len(qids):>5}  "
                f"{diff:>+10.4f}  {ci_str:>20}  {p:>9.4f}"
            )

            results.append({
                "dataset": dataset,
                "method": method,
                "diff": diff,
                "ci_low": ci.low,
                "ci_high": ci.high,
            })

        print()

    return results


def plot_forest(
    results: list[dict],
    plot_output: str | Path | None = None,
) -> None:
    output = Path(plot_output) if plot_output else PLOT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    method_colors = {"hyde": "#1f77b4", "hybrid": "#2ca02c", "hype": "#d62728"}
    method_labels = {"hyde": "HyDE", "hybrid": "Hybridsökning", "hype": "HyPE"}

    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            row = next(
                x for x in results if x["dataset"] == dataset and x["method"] == method
            )
            rows.append(row)
    rows = list(reversed(rows))

    fig, ax = plt.subplots(figsize=(9, 8))

    y_positions = list(range(len(rows)))
    labels = []

    for y, row in zip(y_positions, rows):
        color = method_colors[row["method"]]
        ax.errorbar(
            row["diff"],
            y,
            xerr=[[row["diff"] - row["ci_low"]], [row["ci_high"] - row["diff"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
        labels.append(f"{row['dataset']} — {method_labels[row['method']]}")

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta NDCG@10 (metod - baslinje) med 95 % KI")
    ax.set_title("Parvis jämförelse mot baslinjen per dataset")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=method_colors[method],
            markersize=8,
            label=method_labels[method],
        )
        for method in METHODS
    ]
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"\nForest plot saved to: {output}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_significance_tests()
    plot_forest(results)
