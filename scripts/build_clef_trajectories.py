"""Build labelled, completed CLEF TAR AutoTAR trajectories without the CLEF archives.

Rankings are the hash-pinned AutoTAR files of Bin-Hezam and Stevenson
(ReemBinHezam/TAR_Stopping_Point_Processes at 709f808). Labels for 2018 and 2019
are the official abstract-level qrels of CLEF-TAR/tar at dbc13d0. The 2017
rankings cover 42 topics that mix CLEF's 2017 test and training sets, and were
produced with the per-topic qrels of dli1/auto-stop-tar at 7e72795 (the layout
of the CLEF archives pyterrier-tar downloads); those qrels are used for 2017.
With CLEF-TAR's training qrels instead, seven of the 2017 topics place their
relevant documents at the end of the ranking. A topic's pool is its ranked
documents plus every judged document; judged documents the ranking omits are
appended in qrels order, as pyterrier-tar's ``complete_ranking`` does for
judged documents without text. Unjudged ranked documents are non-relevant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

AUTOTAR_SHA256 = {
    "clef2017": "196b7266a84ecbf491290ac101e7a4952188374c441851ad06bb5d1c431952d2",
    "clef2018": "a108a19b6374ea27f74d2e5eb29c90aaceecf762f8855c83b94fb4e1c4d3d825",
    "clef2019": "2091902efdd08feb66ac6a02f6e37627aea4f906010f3f065aaeb27ba74b49df",
}
RANKINGS = {
    "clef2017": "clef2017_autotar_ranking.txt",
    "clef2017train": "clef2017_training_autotar_ranking.txt",
    "clef2018": "clef2018_autotar_ranking.txt",
    "clef2019": "clef2019_autotar_ranking.txt",
}
# Test qrels first: a document judged in two files takes its first file's label.
# The pool is the union, which matches the archive pools pyterrier-tar reports
# (2019's CD011686 is judged in both files; the training file adds 25533074).
QRELS = {
    # The released 2017 AutoTAR rankings mix CLEF's 2017 test and training topics.
    "clef2017": "autostop",
    "clef2017train": "autostop",
    "clef2018": ["2018-TAR/Task2/Testing/qrels/full.test.abs.2018.qrels"],
    "clef2019": [
        "2019-TAR/Task2/Testing/DTA/qrels/full.test.dta.abs.2019.qrels",
        "2019-TAR/Task2/Testing/Intervention/qrels/full.test.intervention.abs.2019.qrels",
        "2019-TAR/Task2/Testing/Prognosis/qrels/full.test.prognosis.abs.2019.qrels",
        "2019-TAR/Task2/Testing/Qualitative/qrels/full.test.qualitative.abs.2019.qrels",
        "2019-TAR/Task2/Training/DTA/qrels/full.train.dta.abs.2019.qrels",
        "2019-TAR/Task2/Training/Intervention/qrels/full.train.int.abs.2019.qrels",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_qrels(root: Path, files: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(root / name, sep=r"\s+", names=["qid", "iter", "docno", "label"], dtype={"qid": str, "docno": str})
              for name in files]
    qrels = pd.concat(frames, ignore_index=True).drop_duplicates(["qid", "docno"], keep="first")
    qrels["label"] = (qrels.label.astype(int) > 0).astype(int)
    return qrels[["qid", "docno", "label"]]


def read_autostop_qrels(root: Path, topics: set[str]) -> tuple[pd.DataFrame, dict]:
    frames, hashes = [], {}
    for qid in sorted(topics):
        path = root / "data" / "clef2017" / "qrels" / qid
        hashes[qid] = sha256(path)
        frame = pd.read_csv(path, sep=r"\s+", names=["qid", "iter", "docno", "label"], usecols=range(4), dtype={"qid": str, "docno": str})
        frames.append(frame)
    qrels = pd.concat(frames, ignore_index=True).drop_duplicates(["qid", "docno"], keep="first")
    qrels["label"] = (qrels.label.astype(int) > 0).astype(int)
    return qrels[["qid", "docno", "label"]], hashes


def build(collection: str, rankings: Path, qrels_root: Path, autostop_root: Path) -> tuple[pd.DataFrame, dict]:
    path = rankings / RANKINGS[collection]
    if collection in AUTOTAR_SHA256 and sha256(path) != AUTOTAR_SHA256[collection]:
        raise RuntimeError(f"{path} does not match its pinned SHA-256")
    ranked = pd.read_csv(path, sep=r"\s+", names=["qid", "iter", "docno", "rank", "score", "run"], dtype={"qid": str, "docno": str})
    if QRELS[collection] == "autostop":
        qrels, qrels_hashes = read_autostop_qrels(autostop_root, set(ranked.qid))
    else:
        qrels = read_qrels(qrels_root, QRELS[collection])
        qrels_hashes = {name: sha256(qrels_root / name) for name in QRELS[collection]}
    frames, appended = [], {}
    for qid, ranking in ranked.groupby("qid", sort=False):
        ranking = ranking.sort_values("rank", kind="stable").drop_duplicates("docno")
        judged = qrels.loc[qrels.qid == qid]
        if judged.empty:
            raise RuntimeError(f"no qrels for {collection} topic {qid}")
        ranked_docnos = set(ranking.docno)
        tail = [docno for docno in judged.docno if docno not in ranked_docnos]
        if tail:
            appended[qid] = len(tail)
        docnos = list(ranking.docno) + tail
        frame = pd.DataFrame({"qid": qid, "docno": docnos, "rank": range(len(docnos))})
        frame = frame.merge(judged[["docno", "label"]], on="docno", how="left", validate="one_to_one")
        frame["label"] = frame.label.fillna(0).astype(int)
        frames.append(frame)
    trajectory = pd.concat(frames, ignore_index=True)
    sizes = trajectory.groupby("qid").size()
    relevant = trajectory.groupby("qid").label.sum()
    summary = {
        "collection": collection, "topics": int(len(sizes)), "documents": int(sizes.sum()),
        "pool_min_median_max": [int(sizes.min()), float(sizes.median()), int(sizes.max())],
        "relevant_min_median_max": [int(relevant.min()), float(relevant.median()), int(relevant.max())],
        "topics_without_relevant": sorted(relevant.index[relevant == 0]),
        "appended_judged_documents": int(sum(appended.values())), "topics_with_appended": len(appended),
        "ranking_sha256": sha256(path), "qrels_sha256": qrels_hashes,
    }
    return trajectory, summary




def build_runs(pp_root: Path, qrels_root: Path, output: Path) -> list[dict]:
    """Participant runs of 2017 and 2018 as extra training rankings, labelled with CLEF-TAR test qrels.

    Uses pyterrier-tar's pinned run list and its ``read_clef_run`` (file order,
    first listing of a repeat). Documents outside the judged pool are dropped
    and omitted pool documents appended. Runs that omit relevant documents
    (the study's exclusions) and byte-identical duplicates are skipped.
    """
    from pyterrier_tar._clef_files import RUNS
    from pyterrier_tar.datasets import read_clef_run

    qrels_files = {2017: ["2017-TAR/testing/qrels/qrel_abs_test.txt"], 2018: ["2018-TAR/Task2/Testing/qrels/full.test.abs.2018.qrels"]}
    summaries = []
    for year, runs in RUNS.items():
        qrels = read_qrels(qrels_root, qrels_files[year])
        by_topic = {qid: frame for qid, frame in qrels.groupby("qid", sort=False)}
        seen = set()
        for name, expected in sorted(runs.items()):
            path = pp_root / "RunMultipleRankings" / "CLEFData" / "clef_runs" / str(year) / "runs" / name
            digest = sha256(path)
            if digest != expected:
                raise RuntimeError(f"{path} does not match its pinned SHA-256")
            if digest in seen:
                continue
            seen.add(digest)
            ranked = read_clef_run(path)
            frames, omitted_relevant = [], 0
            for qid, ranking in ranked.groupby("qid", sort=False):
                if qid not in by_topic:
                    continue
                judged = by_topic[qid]
                pool = set(judged.docno)
                docnos = [docno for docno in ranking.docno if docno in pool]
                listed = set(docnos)
                tail = [docno for docno in judged.docno if docno not in listed]
                omitted_relevant += int(judged.loc[judged.docno.isin(tail), "label"].sum())
                frame = pd.DataFrame({"qid": qid, "docno": docnos + tail, "rank": range(len(docnos) + len(tail))})
                frames.append(frame.merge(judged[["docno", "label"]], on="docno", how="left", validate="one_to_one"))
            summary = {"year": year, "run": name, "sha256": digest, "topics": len(frames), "omitted_relevant": omitted_relevant}
            summaries.append(summary)
            if omitted_relevant:
                continue
            trajectory = pd.concat(frames, ignore_index=True)
            trajectory["label"] = trajectory.label.fillna(0).astype(int)
            target = output / "runs" / str(year) / (name.replace("/", "__") + ".parquet")
            target.parent.mkdir(parents=True, exist_ok=True)
            trajectory.to_parquet(target, index=False)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rankings", type=Path, help="data/rankings of TAR_Stopping_Point_Processes at 709f808")
    parser.add_argument("qrels_root", type=Path, help="checkout of CLEF-TAR/tar at dbc13d0")
    parser.add_argument("autostop_root", type=Path, help="checkout of dli1/auto-stop-tar at 7e72795 (data/clef2017/qrels)")
    parser.add_argument("--output", type=Path, default=Path("data/clef"))
    parser.add_argument("--runs", action="store_true", help="also build the 2017 and 2018 participant runs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for collection in RANKINGS:
        trajectory, summary = build(collection, args.rankings, args.qrels_root, args.autostop_root)
        trajectory.to_parquet(args.output / f"{collection}.parquet", index=False)
        summaries.append(summary)
        print(json.dumps(summary))
    (args.output / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if args.runs:
        runs = build_runs(args.rankings.parents[1], args.qrels_root, args.output)
        (args.output / "runs_manifest.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(f"{sum(not run['omitted_relevant'] for run in runs)} runs kept of {len(runs)}")



if __name__ == "__main__":
    main()
