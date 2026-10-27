"""AutoTAR-style continuous active learning rankings of the ASReview SYNERGY 1.0 reviews.

SYNERGY ships labels and records but no rankings, so each review is ranked by
simulating AutoTAR (Cormack and Grossman), the procedure behind the CLEF
rankings used for training:

* features: TF-IDF of title and abstract (sublinear term frequency, words
  appearing in at least two records);
* seed: the review's title as a synthetic relevant document;
* each round fits logistic regression on the seed, every reviewed record and
  100 random unreviewed records labelled non-relevant (AutoTAR's
  pseudo-negatives), then reviews the top ``B`` unreviewed records, with ``B``
  starting at 1 and growing by ``ceil(B / 10)``;
* until every record is reviewed.

Records are read with pyterrier-tar's ``SynergyDataset`` (pinned revision,
SHA-256-checked files). Records without any text are reviewed last, as
pyterrier-tar's ``complete_ranking`` appends them. Writes ``synergy.parquet``
(qid, docno, rank, label) and a manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PSEUDO_NEGATIVES = 100


def autotar(texts: list[str], seed_text: str, labels: np.ndarray, seed: int) -> np.ndarray:
    """Review order (indices into ``texts``) of a simulated AutoTAR run."""
    rng = np.random.default_rng(seed)
    vectoriser = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    features = vectoriser.fit_transform(texts)
    seed_vector = vectoriser.transform([seed_text])
    unreviewed = np.ones(len(texts), dtype=bool)
    order: list[int] = []
    batch = 1
    while unreviewed.any():
        reviewed = np.asarray(order, dtype=int)
        pool = np.flatnonzero(unreviewed)
        negatives = rng.choice(pool, size=min(PSEUDO_NEGATIVES, len(pool)), replace=False)
        x = vstack([seed_vector, features[reviewed], features[negatives]])
        y = np.r_[1, labels[reviewed], np.zeros(len(negatives))]
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(x, y)
        scores = model.decision_function(features[pool])
        take = pool[np.argsort(-scores, kind="stable")[:batch]]
        order.extend(int(index) for index in take)
        unreviewed[take] = False
        batch += ceil(batch / 10)
    return np.asarray(order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="pyterrier-tar data home for SYNERGY downloads")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from pyterrier_tar.datasets import SynergyDataset

    dataset = SynergyDataset("1.0")
    dataset._configure(str(args.cache))
    topics = dataset.get_topics()
    qrels = dataset.get_qrels()
    frames, manifest = [], []
    for topic in topics.itertuples():
        started = time.time()
        judged = qrels.loc[qrels.qid == topic.qid]
        records = pd.DataFrame(dataset.get_corpus_iter(topic.qid, verbose=False))
        records["text"] = (records.title.fillna("") + " " + records.text.fillna("")).str.strip()
        records = records.loc[records.text.str.len() > 0].drop_duplicates("docno")
        labelled = records.merge(judged[["docno", "label"]], on="docno", how="inner", validate="one_to_one")
        labels = (labelled.label.to_numpy() > 0).astype(int)
        order = autotar(labelled.text.tolist(), str(topic.title), labels, args.seed)
        ranked = list(labelled.docno.to_numpy()[order])
        tail = [docno for docno in judged.docno if docno not in set(ranked)]
        docnos = ranked + tail
        frame = pd.DataFrame({"qid": topic.qid, "docno": docnos, "rank": range(len(docnos))})
        frame = frame.merge(judged[["docno", "label"]], on="docno", how="left", validate="one_to_one")
        frame["label"] = (frame.label.fillna(0) > 0).astype(int)
        frames.append(frame)
        relevant = frame.label.to_numpy()
        oracle = int(np.flatnonzero(np.cumsum(relevant) >= np.ceil(0.9 * relevant.sum()))[0] + 1) if relevant.any() else 0
        manifest.append({"qid": topic.qid, "pool": len(frame), "relevant": int(relevant.sum()), "without_text": len(tail),
                         "oracle_review_at_0.9": oracle / len(frame), "seconds": round(time.time() - started, 1)})
        print(json.dumps(manifest[-1]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(args.output / "synergy.parquet", index=False)
    (args.output / "synergy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
