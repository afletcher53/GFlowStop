# A stopping flow network that is a GFlowNet where it can be

## Status

This document was written in two stages. The protocol and claim gates below
were committed before any leave-one-collection-out result existed. The results
section was added afterwards and must be read against them.

## The flaw in the original construction, stated as flow

A GFlowNet assumes that every edge of its DAG is chosen by the sampler. It sums
the flow of a state's children, so the policy can put mass on whichever child
is best. In stopping, two kinds of edge are not chosen by the reviewer:

* after `continue`, the environment reveals the next label;
* after a terminal action, the latent world (the topic's yield, or whether the
  target recall was reached) decides the payoff.

The fixed-chain construction summed over those edges too. Its continue flow is
the sum of every terminal reward further along one realised ranking, so the
reviewer is priced as if it could choose its best future stop with the future
labels in hand. `CLAIMS_AND_APPROACH.md` records why that is not
value-of-information training.

## The construction used here

`src/gflowstop/flow.py` keeps the GFlowNet constraints at the reviewer's
(agent) nodes:

    F(h) = sum_a F(h, a),     P_F(a | h) = F(h, a) / F(h),     F(terminal) = R = exp(beta * U),

and makes the aggregation at chance nodes explicit:

| Aggregator | Chance-node constraint | In value units `Q = log F / beta` | Large-`beta` limit |
| --- | --- | --- | --- |
| geometric | `log F(h,a) = E[log F(child) | h]` | `Q(h, continue) = E[V(h') | h]`, `V = (1/beta) logsumexp(beta Q)` | the Bayes-optimal policy |
| arithmetic | `F(h,a) = E[F(child) | h]` (the expected-flow constraint of EFlowNets) | `(1/beta) log E exp(beta V)` | the best outcome, not the expected one |
| hindsight | each realised chain sums its own future terminal rewards | `E[(1/beta) log sum_future exp(beta U)]` | the best future stop with future labels known |

With the geometric aggregator the flows are exactly the soft Bellman equation
of the belief MDP. When no chance node is left (a known world) all three
coincide with an ordinary tree GFlowNet whose terminal distribution is
proportional to `R`. The stopping indicator is the forward policy's
`P_F(stop | h)`.

`tests/test_flow.py` makes these properties executable:

* the exact geometric network attains the Bayes value of `ExactOracle` on the
  beta-binomial and delayed-information toys;
* with a point-mass law, every aggregator samples terminals in proportion to
  reward;
* the arithmetic network stops immediately in the perfect-information toy,
  where the Bayes policy continues;
* the hindsight continue flow is never below the geometric one, and is
  strictly above it somewhere;
* a network learned from realised chains alone reaches the Bayes value on an
  exchangeable toy within 0.002.

### Learning from realised rankings

Nothing about a real collection's law is known, so the network is learned from
realised rankings. A terminal head regresses on realised utilities; the
squared or cross-entropy loss makes it the posterior expectation, which is the
geometric chance node. The continue head cannot regress on a ranking's own
future (that is the hindsight target). Two solvers are implemented:

* `solver='bootstrap'`: one-step soft backup through a frozen copy (fitted soft
  value iteration). Over 200 checkpoints it showed the maximisation bias of
  fitted Q iteration: continue values above every attainable utility.
* `solver='policy'` (used): soft policy iteration. Each round evaluates the
  frozen copy's forward policy exactly along every realised ranking and
  regresses on the value at the next checkpoint. Only the policy's own choices
  are averaged, and they depend only on observed features, so the target
  carries neither bias. Its fixed point satisfies the geometric constraints.

### Target-recall stopping (Track R)

`src/gflowstop/tar_flow.py` applies the network to the TAR task. Stopping
after `n` of `N` documents has utility

    U = -(n/N + penalty * [recall(n) < target]) / (1 + penalty)

The flows are parameterised by that structure:

    log F(x, stop)     / beta = -(n/N + penalty * p_miss(x)) / (1 + penalty)
    log F(x, continue) / beta = -(c(x) + penalty * m(x)) / (1 + penalty) + e(x)

`p_miss` is a classifier of whether the target is still unmet. It never sees
the penalty. `c`, `m` and `e` are the review fraction, miss probability and
entropy bonus still to come under the current forward policy, and `c` is bounded
below by the next checkpoint's review. The target and the penalty are inputs, so
one network covers every operating point. Every feature is computed from the
reviewed prefix only; a test flips every later label and checks that no feature
changes. Stops are allowed at 200 evenly spaced checkpoints (every document for
rankings shorter than 200).

## Data

`scripts/build_clef_trajectories.py` rebuilds the CLEF 2017–2019 AutoTAR
trajectories without the CLEF archives, which the session's network policy
blocked:

* rankings: the hash-pinned AutoTAR files of Bin-Hezam and Stevenson
  (`TAR_Stopping_Point_Processes` at `709f808`, the same SHA-256 values
  pyterrier-tar pins);
* labels for 2018 and 2019: CLEF-TAR abstract qrels at `dbc13d0`;
* labels for 2017: the per-topic qrels of `dli1/auto-stop-tar` at `7e72795`.
  The 2017 ranking file mixes CLEF's 2017 test and training topics, and was
  produced with these labels. With CLEF-TAR's training qrels, seven 2017 topics
  put their relevant documents at the very end of the ranking and the oracle's
  mean review at 0.9 becomes 28%, not 5.7%.

The oracle's mean review at 0.9 is 5.7%, 6.7% and 7.1% for 2017, 2018 and
2019, which is pyterrier-tar's published value for each year. Its baseline replay
(`scripts/clef_baselines.py`) reproduces the estimator and heuristic rows of
pyterrier-tar's `RESULTS.md`.

Extra training rankings: the twelve further 2017 AutoTAR topics
(`clef2017_training_autotar_ranking.txt`) and 48 of the 2017 and 2018
participant runs that Bin-Hezam and Stevenson curated. These are labelled with
the CLEF-TAR test qrels. Runs that omit relevant documents are skipped, as are
byte-identical duplicates.

## Protocol (fixed before the evaluation)

* Leave one collection out. When 2017, 2018 or 2019 is the test collection,
  training uses the AutoTAR rankings of the other two, the twelve extra 2017
  topics, and participant runs of other years (run topics weigh 0.25 each). Any
  ranking whose topic id is in the test collection is dropped (CD011686 is
  judged in both 2018 and 2019).
* The penalty for each target is chosen by inner leave-one-collection-out over
  the two training collections: the cheapest grid penalty whose pooled
  validation reliability is at least 0.95. Test labels are used only to score
  stops.
* Each network is an ensemble of three seeds (flows averaged in log space),
  with `beta = 1e4`, two 64-unit layers and 120 rounds of 25 steps.
* Every rule, learned or not, is scored by pyterrier-tar's `run_experiment` and
  `stopping_metrics` on the same trajectories.

Disclosure: the features, the structured parameterisation, and the switch from
the bootstrap solver to policy iteration were developed while inspecting
failures on the 2019 fold with its labels. Penalty selection never used test
labels. The 2019 row is therefore not a clean held-out estimate; 2017 and 2018
are.

### Claim gates

At each target, a learned rule **beats** a baseline only if both hold:

1. its mean review fraction is lower, with the 95% topic-bootstrap interval of
   the paired difference entirely below zero; and
2. its reliability (share of the 91 topics reaching the target) is at least the
   baseline's.

The primary comparisons are with IP-H (`tail='model'`), the most reliable of
the cheap estimators, with IP-P, which is reliable on every topic, with
ConsecutiveIrrelevant, and with RLStop trained on the same topics. A result that
beats a baseline on review but loses reliability is reported as a trade-off,
not a win. The grid-penalty frontier is descriptive only, because choosing a
point on it uses test labels.

Ablations under the same protocol, with no gate: hindsight continue targets
(the original fixed-chain flow), one-step myopic targets, and `beta = inf`
(hard policy iteration).

## Results of the declared evaluation (revision 1)

Written after the run, against the gates above. Configuration
`results/flow/v1/config.json` (28 base features, the `cheapest` penalty rule,
penalties up to 64). Pooled over the 91 test topics, at the penalty chosen by
inner validation:

| Target | Reliability | Mean review | IP-H (`tail='model'`) reliability / review |
| --- | --- | --- | --- |
| 0.8 | 96.7% | 13.1% | 100.0% / 17.6% |
| 0.9 | 95.6% | 19.7% | 97.8% / 18.5% |
| 0.95 | 95.6% | 32.2% | 97.8% / 20.0% |

It does not beat IP-H (`tail='model'`) at any target. At 0.8 it reviews less but
is less reliable, which the gates count as a trade-off. The chosen penalties
varied erratically between folds: for 2018 the penalty was 64 at target 0.8 but
1 at 0.9. Inner validation picked a lucky low penalty from about sixty
topics.

## Revision 2 (declared before its run)

Changes, decided after seeing revision 1's test rows and recorded here before
revision 2 runs:

1. Tail-fit features: a Poisson power-law rate fitted to the reviewed prefix
   (the IP-P family) with a rough Poisson upper bound, and an exponential rate
   fitted to the last half of the prefix, each as an estimated recall and its
   log-odds margin from the target.
2. Penalty selection `stable`: the smallest penalty from which pooled
   validation reliability stays at or above 0.95 for every larger penalty.
3. Penalty grid and training range extended to 256.

Everything else, including the gates, is unchanged. Because these changes
followed revision 1's test results, revision 2 is a second look at the same
91 topics and not an independent confirmation; a held-out confirmation would
need another collection.
