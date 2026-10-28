# Data for the stopping flow network experiments

`toys/` holds the fixed synthetic episodes of `gflowstop.toy_experiment`.

## `clef/`

Labelled, completed CLEF TAR AutoTAR trajectories (`qid`, `docno`, `rank`,
`label`), one file per collection. Every stopping rule in
`GFLOWNET_STOPPING.md` was scored on these:

* `clef2017`, `clef2018`, `clef2019`: the 30, 30 and 31 test topics;
* `clef2017train`: 12 further 2017 topics, used only for training.

They were built by `scripts/build_clef_trajectories.py` from:

* the hash-pinned AutoTAR rankings of `ReemBinHezam/TAR_Stopping_Point_Processes`
  at `709f808`;
* CLEF-TAR abstract qrels at `dbc13d0` for 2018 and 2019;
* the per-topic qrels of `dli1/auto-stop-tar` at `7e72795` for 2017, which the
  2017 rankings match.

`manifest.json` records the SHA-256 of every input. The 48 participant-run
trajectories used for training (about 80 MB) are not committed. Rebuild them
with the same script and `--runs`; `runs_manifest.json` lists them with their
hashes.

## `synergy/`

`synergy.parquet` holds the 26 ASReview SYNERGY 1.0 reviews, ranked by the
AutoTAR simulation in `scripts/build_synergy_rankings.py` (seed 0) with
final-inclusion labels. `synergy_manifest.json` gives each review's pool,
inclusions and oracle review at recall 0.9. The records themselves are not
redistributed; pyterrier-tar's `SynergyDataset` fetches them from the pinned
SYNERGY revision.

Trained networks and RLStop policies for the reported runs are in
`results/flow/models/`. They are Python pickles and stable-baselines3 archives,
so load them only from this repository.
