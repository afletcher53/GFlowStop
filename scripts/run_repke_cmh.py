"""Run the public framework's CMH (BUSCAR with bias=1) on fixed rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

import pandas as pd

from gflowstop.external_benchmark import (
    _checkpoints,
    _first_stops,
    _summary,
    _sha256,
    load_repke_rankings,
)


def _trace_one(args: tuple[object, ...]) -> list[dict[str, object]]:
    framework_root, ranking, target, confidence, check_interval = args
    framework_root = Path(framework_root)
    if str(framework_root) not in sys.path:
        sys.path.insert(0, str(framework_root))
    # This is the published implementation, not a reimplementation.
    from methods.buscar import Buscar

    method_hash = hashlib.sha256(
        json.dumps({"method": "CMH", "target": target, "confidence": confidence}, sort_keys=True).encode()
    ).hexdigest()
    rows: list[dict[str, object]] = []
    labels = ranking.labels
    previous = 0
    for batch_i, depth in enumerate(_checkpoints(len(labels), check_interval)):
        result = Buscar.compute(
            n_total=len(labels), labels=labels[:depth], recall_target=target,
            confidence_level=confidence, bias=1.0,
        )
        hits = int(labels[:depth].sum())
        recall = 1.0 if not labels.sum() else float(hits / labels.sum())
        rows.append({
            "dataset": ranking.dataset,
            "sim-rep": ranking.simulation,
            "sim_key": ranking.key,
            "fold": 0,
            "batch_i": batch_i,
            "n_total": len(labels),
            "n_seen": depth,
            "n_unseen": len(labels) - depth,
            "n_incl": int(labels.sum()),
            "n_incl_seen": hits,
            "n_records_batch": depth - previous,
            "method": "CMH",
            "method-KEY": "BUSCAR",
            "method-hash": method_hash,
            "method-recall_target": target,
            "method-confidence_level": confidence,
            "method-bias": 1.0,
            "method-score": None if result["score"] is None else float(result["score"]),
            "safe_to_stop": bool(result["safe_to_stop"]),
            "realised_recall": recall,
            "recall_oracle_depth": int(next(
                (index + 1 for index, value in enumerate(labels.cumsum())
                 if value >= ceil(target * int(labels.sum()) - 1e-12)),
                0,
            )),
            "info_path": str(ranking.info_path),
            "ranking_path": str(ranking.ranking_path),
        })
        previous = depth
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking_dir", type=Path)
    parser.add_argument("--framework-root", type=Path, required=True)
    parser.add_argument("--framework-revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/repke_cmh"))
    parser.add_argument("--targets", default="0.8,0.9,0.95")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--check-interval", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    revision = subprocess.run(
        ["git", "-C", args.framework_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != args.framework_revision:
        raise ValueError(f"framework revision mismatch: expected {args.framework_revision}, got {revision}")
    rankings = load_repke_rankings(args.ranking_dir)
    targets = tuple(float(value) for value in args.targets.split(",") if value)
    tasks = [(args.framework_root, ranking, target, args.confidence, args.check_interval)
             for ranking in rankings for target in targets]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = [row for trace in executor.map(_trace_one, tasks) for row in trace]
    trace = pd.DataFrame(rows)
    first_stops = _first_stops(trace)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(output_dir / "repke_trace.csv", index=False)
    first_stops.to_csv(output_dir / "repke_first_stops.csv", index=False)
    _summary(first_stops).to_csv(output_dir / "repke_summary.csv", index=False)
    metadata = {
        "gflowstop_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "gflowstop_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
        ).stdout.strip()),
        "framework_root": str(args.framework_root.resolve()),
        "framework_revision": revision,
        "framework_method": "methods.buscar.Buscar (bias=1.0; CMH)",
        "ranking_dir": str(args.ranking_dir.resolve()),
        "selected_rankings": len(rankings),
        "selected_datasets": len({ranking.dataset for ranking in rankings}),
        "targets": targets,
        "confidence_level": args.confidence,
        "check_interval": args.check_interval,
        "ranking_files": {str(ranking.ranking_path): _sha256(ranking.ranking_path) for ranking in rankings},
        "info_files": {str(ranking.info_path): _sha256(ranking.info_path) for ranking in rankings},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
