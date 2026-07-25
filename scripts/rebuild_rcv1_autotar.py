#!/usr/bin/env python3
"""Build a provenance-locked RCV1 AutoTAR-style ranking reproduction.

The published RCV1 AutoTAR run is unavailable.  This builder deliberately
creates a new realization: a fixed 20% document universe, the public 45-topic
test manifest, and a bounded seeded active-learning trajectory.  It must not
be described as the unpublished GRLStop ranking.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Avoid multiplying BLAS workers when several topic trajectories run together.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DOCUMENT_COUNT = 804_414
SUBSET_SIZE = 160_882
SOURCE_ROOT = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files"
RAW_FILES = {
    "topics": "rcv1-v2.topics.qrels.gz",
    "industries": "rcv1-v2.industries.qrels.gz",
    "regions": "rcv1-v2.regions.qrels.gz",
}
TEST_CATEGORIES = tuple(
    """
    regions_ASIA industries_I35102 industries_I65100 industries_I41300 industries_I32200
    industries_I24000 industries_I32830 industries_I42600 industries_I8150211 regions_RWANDA
    regions_ALG regions_GUAT regions_BURMA regions_CUBA regions_MCDNIA topics_GENV
    industries_I3302021 topics_C182 topics_GPRO industries_I82000 industries_I22100 topics_E513
    industries_I36400 industries_I42900 industries_I41000 regions_FIN regions_CZREP regions_BUL
    regions_SWED regions_TAIWAN industries_I83100 topics_E12 industries_I81501 topics_C33
    industries_I81502 topics_C12 regions_BELG industries_I21000 industries_I1300003 industries_I14000
    regions_INDON regions_MEX regions_ISRAEL regions_POL regions_EEC
    """.split()
)
LSR_MANIFEST_URL = "https://raw.githubusercontent.com/eugene-yang/LSR-for-TAR/233b75915a8f00698ad7882929ff5919af25983a/rcv1_cates.txt"
AUTOSTOP_REVISION = "7e727950cef8f9eea553cd2b56068141ccf01713"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_documents(paths: list[Path]):
    for path in paths:
        did: str | None = None
        text: list[str] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if line.startswith(".I "):
                    if did is not None:
                        yield did, " ".join(text)
                    did, text = line[3:].strip(), []
                elif line == ".W":
                    continue
                elif line:
                    text.append(line)
                elif did is not None:
                    yield did, " ".join(text)
                    did, text = None, []
        if did is not None:
            yield did, " ".join(text)


def _selected_documents(raw_dir: Path, seed: int) -> pd.DataFrame:
    positions = set(np.random.default_rng(seed).choice(DOCUMENT_COUNT, SUBSET_SIZE, replace=False).tolist())
    paths = [raw_dir / f"lyrl2004_tokens_test_pt{part}.dat.gz" for part in range(4)]
    paths.append(raw_dir / "lyrl2004_tokens_train.dat.gz")
    selected: list[tuple[str, str]] = []
    total = 0
    for total, record in enumerate(_token_documents(paths), start=1):
        if total - 1 in positions:
            selected.append(record)
    if total != DOCUMENT_COUNT or len(selected) != SUBSET_SIZE:
        raise ValueError(f"token source mismatch: saw {total} documents and selected {len(selected)}")
    return pd.DataFrame(selected, columns=["doc_id", "text"])


def _read_qrels(path: Path, namespace: str, selected: set[str], wanted: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    labels = {category: set() for category in wanted if category.startswith(f"{namespace}_")}
    training_topics: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            category, did, _ = line.split()
            if namespace == "topics" and int(did) <= 26_150:
                training_topics.add(category)
            key = f"{namespace}_{category}"
            if key in labels and did in selected:
                labels[key].add(did)
    return labels, training_topics


def prepare(root: Path, seed: int, overwrite: bool) -> None:
    raw_dir = root / "raw"
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    documents_path = prepared / "documents.parquet"
    targets_path = prepared / "targets.npz"
    manifest_path = prepared / "manifest.json"
    outputs = (documents_path, targets_path, manifest_path)
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError("prepared RCV1 artifacts already exist; pass --overwrite to rebuild")
    missing = [raw_dir / filename for filename in RAW_FILES.values()]
    missing += [raw_dir / f"lyrl2004_tokens_test_pt{part}.dat.gz" for part in range(4)]
    missing += [raw_dir / "lyrl2004_tokens_train.dat.gz"]
    if absent := [str(path) for path in missing if not path.exists()]:
        raise FileNotFoundError(f"missing raw RCV1 inputs: {absent}")

    documents = _selected_documents(raw_dir, seed)
    selected = set(documents.doc_id)
    test_categories = set(TEST_CATEGORIES)
    grouped: dict[str, dict[str, set[str]]] = {}
    trainable_topics: set[str] = set()
    for namespace, filename in RAW_FILES.items():
        labels, topic_names = _read_qrels(raw_dir / filename, namespace, selected, test_categories)
        grouped[namespace] = labels
        trainable_topics |= topic_names

    test_topic_codes = {category.removeprefix("topics_") for category in TEST_CATEGORIES if category.startswith("topics_")}
    train_categories = tuple(f"topics_{category}" for category in sorted(trainable_topics - test_topic_codes))
    if len(TEST_CATEGORIES) != 45 or len(train_categories) != 94:
        raise AssertionError(f"expected the RCV1 94/45 partition, got {len(train_categories)}/{len(TEST_CATEGORIES)}")
    categories = train_categories + TEST_CATEGORIES
    positive_ids = {category: grouped.get(category.split("_", 1)[0], {}).get(category, set()) for category in TEST_CATEGORIES}
    for category in train_categories:
        positive_ids[category] = set()
    with gzip.open(raw_dir / RAW_FILES["topics"], "rt", encoding="utf-8") as handle:
        wanted_train = set(train_categories)
        for line in handle:
            category, did, _ = line.split()
            key = f"topics_{category}"
            if key in wanted_train and did in selected:
                positive_ids[key].add(did)
    labels = np.asarray([[did in positive_ids[category] for did in documents.doc_id] for category in categories], dtype=np.bool_)
    if np.any(labels.sum(axis=1) == 0):
        empty = [category for category, count in zip(categories, labels.sum(axis=1)) if not count]
        raise AssertionError(f"downsampled category has no positives: {empty}")

    documents.to_parquet(documents_path, index=False)
    np.savez_compressed(targets_path, categories=np.asarray(categories), labels=labels)
    manifest = {
        "kind": "new RCV1 AutoTAR-style reproduction; not the unpublished GRLStop realization",
        "source_root": SOURCE_ROOT,
        "raw_sha256": {path.name: _sha256(path) for path in sorted(raw_dir.iterdir()) if path.is_file()},
        "test_manifest_source": LSR_MANIFEST_URL,
        "test_manifest_categories": TEST_CATEGORIES,
        "test_manifest_sha256": hashlib.sha256(("\n".join(TEST_CATEGORIES) + "\n").encode()).hexdigest(),
        "autostop_reference_revision": AUTOSTOP_REVISION,
        "document_count": DOCUMENT_COUNT,
        "subset_size": SUBSET_SIZE,
        "subset_fraction": 0.2,
        "sampling": "np.random.default_rng(seed).choice(token-stream positions, subset_size, replace=False)",
        "sampling_seed": seed,
        "training_category_rule": "the 101 topics with a standard LYRL2004 training example, excluding the seven topic labels in the 45-category test manifest",
        "training_categories": train_categories,
        "test_categories": TEST_CATEGORIES,
        "documents_sha256": _sha256(documents_path),
        "targets_sha256": _sha256(targets_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Prepared {len(documents)} documents, {len(train_categories)} training categories, and {len(TEST_CATEGORIES)} test categories in {prepared}")


def _topic_seed(global_seed: int, category: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{global_seed}:{category}".encode()).digest()[:8], "little")


def _rank_topic(features: sparse.csr_matrix, labels: np.ndarray, category: str, seed: int, rounds: int, temporary_negatives: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_topic_seed(seed, category))
    positives = np.flatnonzero(labels)
    if not len(positives):
        raise ValueError(f"{category} has no positive seed candidate")
    assessed = np.zeros(len(labels), dtype=bool)
    assessed[positives[rng.integers(len(positives))]] = True
    batch_size = 1
    scores = np.zeros(len(labels), dtype=np.float64)
    for _ in range(rounds):
        unassessed = np.flatnonzero(~assessed)
        temporary = rng.choice(unassessed, size=min(temporary_negatives, len(unassessed)), replace=False)
        train_indices = np.concatenate((np.flatnonzero(assessed), temporary))
        train_labels = np.concatenate((labels[assessed], np.zeros(len(temporary), dtype=bool)))
        model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=10_000, random_state=int(rng.integers(2**31 - 1)))
        model.fit(features[train_indices], train_labels)
        scores = model.predict_proba(features)[:, 1]
        ordered = np.argsort(-scores, kind="stable")
        choose = ordered[~assessed[ordered]][:batch_size]
        assessed[choose] = True
        batch_size += int(np.ceil(batch_size / 10.0))
    return np.argsort(-scores, kind="stable").astype(np.int32), scores.astype(np.float32)


def rank(root: Path, seed: int, rounds: int, temporary_negatives: int, workers: int, selected_categories: set[str] | None, overwrite: bool) -> None:
    prepared = root / "prepared"
    documents = pd.read_parquet(prepared / "documents.parquet")
    source = np.load(prepared / "targets.npz", allow_pickle=False)
    categories = tuple(str(value) for value in source["categories"])
    labels = source["labels"]
    if labels.shape != (len(categories), len(documents)):
        raise ValueError("target matrix does not match prepared document universe")
    wanted = set(categories) if selected_categories is None else selected_categories
    unknown = wanted - set(categories)
    if unknown:
        raise ValueError(f"unknown RCV1 categories: {sorted(unknown)}")
    ranks = root / "ranks"
    ranks.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, dtype=np.float64)
    features = vectorizer.fit_transform(documents.text).tocsr()
    print(f"Fitted TF-IDF for {features.shape[0]} documents and {features.shape[1]} features")

    todo = [(index, category) for index, category in enumerate(categories) if category in wanted and (overwrite or not (ranks / f"{category}.npz").exists())]
    if not todo:
        print("No topic rankings need rebuilding")
        return

    def one(item: tuple[int, str]) -> str:
        index, category = item
        order, scores = _rank_topic(features, labels[index], category, seed, rounds, temporary_negatives)
        np.savez_compressed(ranks / f"{category}.npz", order=order, scores=scores)
        return category

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in todo]
        for completed, future in enumerate(as_completed(futures), start=1):
            print(f"[{completed}/{len(todo)}] ranked {future.result()}", flush=True)


def _write_csv(path: Path, categories: tuple[str, ...], category_indices: list[int], doc_ids: np.ndarray, labels: np.ndarray, ranks: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Topic", "Rank", "Score", "label"))
        for index in category_indices:
            category = categories[index]
            data = np.load(ranks / f"{category}.npz", allow_pickle=False)
            order, scores = data["order"], data["scores"]
            if len(order) != len(doc_ids) or np.unique(order).size != len(order):
                raise ValueError(f"invalid ranking permutation for {category}")
            writer.writerows((category, rank, float(scores[doc_index]), int(labels[index, doc_index])) for rank, doc_index in enumerate(order, start=1))


def merge(root: Path, overwrite: bool) -> None:
    prepared, ranks = root / "prepared", root / "ranks"
    documents = pd.read_parquet(prepared / "documents.parquet")
    source = np.load(prepared / "targets.npz", allow_pickle=False)
    categories = tuple(str(value) for value in source["categories"])
    labels = source["labels"]
    manifest = json.loads((prepared / "manifest.json").read_text())
    train = set(manifest["training_categories"])
    test = set(manifest["test_categories"])
    missing = [category for category in categories if not (ranks / f"{category}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} topic rankings; first: {missing[:5]}")
    outputs = (prepared / "train_rankings.csv", prepared / "test_rankings.csv")
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError("merged ranking CSVs exist; pass --overwrite to replace them")
    train_indices = [index for index, category in enumerate(categories) if category in train]
    test_indices = [index for index, category in enumerate(categories) if category in test]
    _write_csv(outputs[0], categories, train_indices, documents.doc_id.to_numpy(), labels, ranks)
    _write_csv(outputs[1], categories, test_indices, documents.doc_id.to_numpy(), labels, ranks)
    config_path = root / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError("run_config.json is required to record the ranking trajectory")
    config = json.loads(config_path.read_text())
    manifest["ranking_builder"] = {
        "algorithm": "bounded seeded AutoTAR-style active learning; final logistic-regression scores define the fixed ranking",
        **config,
        "train_csv_sha256": _sha256(outputs[0]),
        "test_csv_sha256": _sha256(outputs[1]),
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {outputs[0]} and {outputs[1]}")


def self_test() -> None:
    features = sparse.csr_matrix(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]))
    labels = np.array([True, True, False, False])
    order, scores = _rank_topic(features, labels, "test", 7, rounds=3, temporary_negatives=2)
    assert set(order) == {0, 1, 2, 3}
    assert np.all(np.diff(scores[order]) <= 0.0)
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "rank", "merge", "self-test"))
    parser.add_argument("--root", type=Path, default=Path("data/external/rcv1_rebuild"))
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--rounds", type=int, default=35)
    parser.add_argument("--temporary-negatives", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topics", help="comma-separated category IDs for a pilot ranking run")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    if args.rounds < 1 or args.temporary_negatives < 1 or args.workers < 1:
        raise ValueError("rounds, temporary negatives, and workers must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command in {"prepare", "rank"}:
        (args.root / "run_config.json").write_text(json.dumps({"seed": args.seed, "rounds": args.rounds, "temporary_negatives": args.temporary_negatives, "workers": args.workers}, indent=2) + "\n")
    if args.command == "prepare":
        prepare(args.root, args.seed, args.overwrite)
    elif args.command == "rank":
        topics = None if args.topics is None else {topic for topic in args.topics.split(",") if topic}
        rank(args.root, args.seed, args.rounds, args.temporary_negatives, args.workers, topics, args.overwrite)
    else:
        merge(args.root, args.overwrite)


if __name__ == "__main__":
    main()
