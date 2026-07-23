# Fixed synthetic stopping benchmark

Track U reports net decision utility; Track R reports recall safety and effort separately. Rows are condition-specific and use fixed saved episodes shared by all methods.

## Utility summary

suite,environment,condition,method,topics,mean_utility,utility_ci_low,utility_ci_high,mean_stop_depth
clustered,clustered,matched_hmm,Rank-score-latent-full,1000,0.1822,0.17674,0.18732049999999997,3.065
clustered,clustered,matched_hmm,Count-Bellman-fitted,1000,0.12949999999999998,0.12021849999999996,0.13795999999999997,4.25
clustered,clustered,matched_hmm,Full-screen,1000,0.05,0.05,0.05,10.0
delayed_information,delayed_information,lookahead=1,Exact-lookahead-1,1000,0.001,-0.0145,0.016,0.0
delayed_information,delayed_information,lookahead=10,Exact-lookahead-10,1000,0.17965,0.17714975,0.18184,6.785
delayed_information,delayed_information,lookahead=2,Exact-lookahead-2,1000,0.001,-0.015,0.016512499999999933,0.0
delayed_information,delayed_information,lookahead=5,Exact-lookahead-5,1000,0.17965,0.17711975,0.18187025,6.785
matched_beta,beta_binomial,matched,Count-Bellman-fitted,5000,0.11193999999999998,0.10854669999999998,0.115248,5.353
matched_beta,beta_binomial,matched,Count-Bellman-true,5000,0.11171999999999999,0.10796389999999999,0.11551209999999999,4.864
matched_beta,beta_binomial,matched,Amortized-count,5000,0.11068399999999999,0.10758780000000001,0.11364490000000001,5.7208
matched_beta,beta_binomial,matched,Hit-probability-cutoff,5000,0.08653999999999999,0.0856557,0.0874041,8.173
matched_beta,beta_binomial,matched,Fixed-budget-half,5000,0.07597999999999999,0.0710796,0.0810801,4.951
matched_beta,beta_binomial,matched,Full-screen,5000,0.04999999999999999,0.04999999999999999,0.04999999999999999,10.0
matched_beta,beta_binomial,matched,Zero-screen,5000,0.0426,0.0358975,0.0494,0.0
mixture_shift,mixture_shift,shifted_test,Rank-score-latent-full,1000,0.19884000000000002,0.18967650000000003,0.20751999999999998,0.308
mixture_shift,mixture_shift,shifted_test,Count-Bellman-fitted,1000,0.1945,0.184,0.2035,0.0
mixture_shift,mixture_shift,shifted_test,Full-screen,1000,0.05,0.05,0.05,10.0
perfect_information,perfect_information,cost=0.01,Bayes-exact,1000,0.2400000000000001,0.2400000000000001,0.2400000000000001,1.0
rank_decay,rank_decay,gamma=0.0,True-rank-oracle,1000,0.24506,0.24233949999999999,0.2472405,0.122
rank_decay,rank_decay,gamma=0.0,Count-Bellman-fitted,1000,0.244,0.2405,0.247,0.0
rank_decay,rank_decay,gamma=0.0,Rank-only-full,1000,0.244,0.2405,0.247,0.0
rank_decay,rank_decay,gamma=0.0,Rank-score-real-full,1000,0.244,0.2405,0.247,0.0
rank_decay,rank_decay,gamma=0.0,Rank-score-shuffled-across-full,1000,0.244,0.2405,0.247,0.0
rank_decay,rank_decay,gamma=0.0,Rank-score-shuffled-within-full,1000,0.244,0.2405,0.247,0.0
rank_decay,rank_decay,gamma=0.5,True-rank-oracle,1000,0.2266,0.220559,0.23178,0.495
rank_decay,rank_decay,gamma=0.5,Rank-score-real-full,1000,0.22348,0.2170735,0.22966049999999996,0.251
rank_decay,rank_decay,gamma=0.5,Rank-score-shuffled-within-full,1000,0.22314,0.2168195,0.2294405,0.243
rank_decay,rank_decay,gamma=0.5,Count-Bellman-fitted,1000,0.2175,0.21,0.225,0.0
rank_decay,rank_decay,gamma=0.5,Rank-only-full,1000,0.2175,0.21,0.225,0.0
rank_decay,rank_decay,gamma=0.5,Rank-score-shuffled-across-full,1000,0.2175,0.2095,0.225,0.0
rank_decay,rank_decay,gamma=1.5,True-rank-oracle,1000,0.17586,0.16837850000000001,0.18288049999999997,2.532
rank_decay,rank_decay,gamma=1.5,Rank-score-shuffled-within-full,1000,0.15824,0.15224,0.16404099999999996,3.963
rank_decay,rank_decay,gamma=1.5,Rank-score-real-full,1000,0.15775999999999998,0.15225949999999996,0.16312049999999997,4.037
rank_decay,rank_decay,gamma=1.5,Rank-only-full,1000,0.14855999999999997,0.14398,0.15253999999999995,4.897
rank_decay,rank_decay,gamma=1.5,Rank-score-shuffled-across-full,1000,0.14749999999999996,0.141819,0.15328149999999996,4.475
rank_decay,rank_decay,gamma=1.5,Count-Bellman-fitted,1000,0.10947999999999998,0.09895999999999998,0.11976049999999999,3.401
rank_decay,rank_decay,gamma=3.0,True-rank-oracle,1000,0.21869999999999998,0.21265899999999996,0.22426049999999997,0.79
rank_decay,rank_decay,gamma=3.0,Rank-score-shuffled-within-full,1000,0.21459999999999996,0.2080995,0.22018049999999997,0.995
rank_decay,rank_decay,gamma=3.0,Rank-score-real-full,1000,0.21427999999999997,0.20844,0.21948199999999995,1.036
rank_decay,rank_decay,gamma=3.0,Rank-score-shuffled-across-full,1000,0.2037,0.19651849999999998,0.2096805,1.24
rank_decay,rank_decay,gamma=3.0,Rank-only-full,1000,0.20316000000000004,0.197939,0.20780200000000001,1.767
rank_decay,rank_decay,gamma=3.0,Count-Bellman-fitted,1000,0.203,0.1939875,0.212,0.0

## Planner summary

method,states,high_gap_action_agreement,mean_planning_regret,maximum_planning_regret
Exact-lookahead-1,152,0.6842105263157895,0.016387259800086127,0.08251014492753617
Exact-lookahead-10,152,1.0,0.0,0.0
Exact-lookahead-2,152,0.8486842105263158,0.003458861725211234,0.02500676328502416
Exact-lookahead-5,152,1.0,0.0,0.0

## Recall summary

suite,environment,condition,method,target_recall,alpha,claims_safety,topics,failure_rate,failure_ci_low,failure_ci_high,reliability_gap,mean_review_fraction,mean_excess_effort,mean_overshoot
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.8,0.05,False,10000,0.0,0.0,0.0,-0.05,0.796407,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.8,0.1,False,10000,0.0,0.0,0.0,-0.1,0.796407,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.9,0.05,False,10000,0.0,0.0,0.0,-0.05,0.8958455,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.9,0.1,False,10000,0.0,0.0,0.0,-0.1,0.8958455,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.95,0.05,False,10000,0.0,0.0,0.0,-0.05,0.9453744999999999,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.95,0.1,False,10000,0.0,0.0,0.0,-0.1,0.9453744999999999,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.8,0.1,True,10000,0.0,0.0,0.0,-0.1,0.9646394999999999,33.6465,0.16470599999999994
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.8,0.05,True,10000,0.0,0.0,0.0,-0.05,0.9648700000000001,33.6926,0.16478399999999993
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.9,0.1,True,10000,0.0,0.0,0.0,-0.1,0.9709615000000001,15.0232,0.07117799999999995
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.9,0.05,True,10000,0.0,0.0,0.0,-0.05,0.971245,15.0799,0.07138999999999995
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.95,0.1,False,10000,0.0,0.0,0.0,-0.1,0.9731079999999999,5.5467,0.02313200000000002
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.95,0.05,False,10000,0.0,0.0,0.0,-0.05,0.9780785,6.5408,0.028062000000000024
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.95,0.05,True,10000,0.0,0.0,0.0,-0.05,0.9933965,9.6044,0.04339200000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.95,0.1,True,10000,0.0,0.0,0.0,-0.1,0.9934755000000001,9.6202,0.043459000000000046
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.8,0.05,True,10000,0.0,0.0,0.0,-0.05,1.0,40.7186,0.19999999999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.8,0.1,True,10000,0.0,0.0,0.0,-0.1,1.0,40.7186,0.19999999999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.9,0.05,True,10000,0.0,0.0,0.0,-0.05,1.0,20.8309,0.09999999999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.9,0.1,True,10000,0.0,0.0,0.0,-0.1,1.0,20.8309,0.09999999999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.95,0.05,True,10000,0.0,0.0,0.0,-0.05,1.0,10.9251,0.050000000000000044
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.95,0.1,True,10000,0.0,0.0,0.0,-0.1,1.0,10.9251,0.050000000000000044
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.8,0.05,False,10000,0.0001,0.0,0.0003,-0.0499,0.9999845000000002,40.7155,0.19997799999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.8,0.1,False,10000,0.0001,0.0,0.0003,-0.0999,0.9999845000000002,40.7155,0.19997799999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.9,0.05,False,10000,0.0001,0.0,0.0003,-0.0499,0.9999845000000002,20.8278,0.09997799999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.9,0.1,False,10000,0.0001,0.0,0.0003,-0.0999,0.9999845000000002,20.8278,0.09997799999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.95,0.05,False,10000,0.0001,0.0,0.0003,-0.0499,0.9999845000000002,10.922,0.04997800000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.95,0.1,False,10000,0.0001,0.0,0.0003,-0.0999,0.9999845000000002,10.922,0.04997800000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.95,0.05,True,10000,0.0023,0.0014975000000000001,0.0033,-0.047700000000000006,0.9934689999999998,9.6189,0.04351900000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.95,0.1,True,10000,0.0034,0.0023,0.0046,-0.0966,0.9933774999999999,9.6006,0.04344300000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.9,0.05,False,10000,0.004,0.0028,0.0053,-0.046,0.9394020000000001,8.7113,0.03896599999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.8,0.1,True,10000,0.0061,0.0046,0.0077,-0.09390000000000001,0.9327235,27.2633,0.13278299999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.8,0.05,True,10000,0.0077,0.0060975000000000005,0.0095,-0.042300000000000004,0.9318445,27.0875,0.13172799999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.9,0.1,False,10000,0.0114,0.0095,0.0135,-0.08860000000000001,0.931334,7.0977,0.03113399999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.9,0.05,True,10000,0.0155,0.0131,0.0179,-0.0345,0.9774164999999999,16.3142,0.07748499999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.9,0.1,True,10000,0.0155,0.0131,0.0179,-0.0845,0.9774305,16.317,0.07738099999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.8,0.05,False,10000,0.0425,0.0385,0.0467,-0.0075,0.846842,10.087,0.04658699999999994
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.8,0.1,False,10000,0.0819,0.0765,0.0875,-0.018100000000000005,0.8358774999999999,7.8941,0.03550699999999995
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QuantCI-estimator,0.8,0.1,False,10000,1.0,1.0,1.0,0.9,0.4433665,-70.6081,-0.35562400000000005
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QuantCI-estimator,0.8,0.05,False,10000,1.0,1.0,1.0,0.95,0.462634,-66.7546,-0.33634100000000006
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QuantCI-estimator,0.9,0.1,False,10000,1.0,1.0,1.0,0.9,0.6378175,-51.6056,-0.261226
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QuantCI-estimator,0.9,0.05,False,10000,1.0,1.0,1.0,0.95,0.66051,-47.0671,-0.23886700000000005
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QuantCI-estimator,0.95,0.1,False,10000,1.0,1.0,1.0,0.9,0.793,-30.4749,-0.15665799999999994
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QuantCI-estimator,0.95,0.05,False,10000,1.0,1.0,1.0,0.95,0.817724,-25.5301,-0.13237599999999997
sparse_recall,sparse_recall,"r=5,gamma=2.0",Recall-oracle,0.9,0.05,False,2000,0.0,0.0,0.0,-0.05,0.6742149999999999,0.0,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Full-screen,0.9,0.05,True,2000,0.0,0.0,0.0,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Point-process-exponential,0.9,0.05,False,2000,0.0,0.0,0.0,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",QBCB,0.9,0.05,True,2000,0.0,0.0,0.0,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",QuantCI-estimator,0.9,0.05,False,2000,0.0,0.0,0.0,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Sequential-hypergeometric,0.9,0.05,True,2000,0.0,0.0,0.0,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Consecutive-miss-20,0.9,0.05,False,2000,1.0,1.0,1.0,0.95,0.04992,-312.1475,-0.7985

## Gates

```json
{
  "A_oracle_correctness": {
    "perfect_information_test": "pytest test_hindsight_flow_target_is_not_bayes_optimal",
    "status": "pass"
  },
  "B_matched_model": {
    "status": "reported",
    "note": "compare true and fitted Count-Bellman learning curve; no post-test tuning"
  },
  "C_rank_aware": {
    "status": "reported",
    "gamma_1_5_mean_utility": {
      "Count-Bellman-fitted": 0.10948000000000001,
      "Rank-only-full": 0.14856,
      "Rank-score-real-full": 0.15775999999999998,
      "Rank-score-shuffled-across-full": 0.1475,
      "Rank-score-shuffled-within-full": 0.15824000000000002,
      "True-rank-oracle": 0.17586
    },
    "next_hit_calibration": {
      "Rank-only": {
        "brier": 0.15286407324625695,
        "log_loss": 0.4719767299396135,
        "ece": 0.03198634731278307
      },
      "Rank-score-real": {
        "brier": 0.14385710042003844,
        "log_loss": 0.4492835228371562,
        "ece": 0.04189588077641967
      },
      "Rank-score-shuffled-within": {
        "brier": 0.14426154094456214,
        "log_loss": 0.45104257567775585,
        "ece": 0.044412196743429125
      },
      "Rank-score-shuffled-across": {
        "brier": 0.15251698218675033,
        "log_loss": 0.47142872441587025,
        "ece": 0.02258432120854458
      }
    }
  },
  "D_planner": {
    "status": "reported",
    "note": "small-POMDP rows separate full and truncated exact planning"
  },
  "E_recall": {
    "status": "reported",
    "claimed_safety_upper_failure": {
      "Full-screen": 0.0,
      "QBCB": 0.00814516129032258,
      "Sequential-hypergeometric": 0.0
    }
  }
}
```
