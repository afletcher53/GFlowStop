# Fixed synthetic stopping benchmark

Track U reports net decision utility; Track R reports recall safety and effort separately. Rows are condition-specific and use fixed saved episodes shared by all methods.

## Utility summary

suite,environment,condition,method,training_topics,topics,mean_utility,utility_ci_low,utility_ci_high,mean_stop_depth,oracle_root_continue_fraction,mean_oracle_root_stop_value,mean_oracle_root_information_value
clustered,clustered,matched_hmm,Rank-score-latent-full,,1000,0.1822,0.17674,0.18732049999999997,3.065,,,
clustered,clustered,matched_hmm,Count-Bellman-fitted,,1000,0.12949999999999998,0.12021849999999996,0.13795999999999997,4.25,,,
clustered,clustered,matched_hmm,Full-screen,,1000,0.05,0.05,0.05,10.0,,,
delayed_information,delayed_information,lookahead=1,Exact-lookahead-1,,1000,0.001,-0.0145,0.016,0.0,,,
delayed_information,delayed_information,lookahead=10,Exact-lookahead-10,,1000,0.17965,0.17714975,0.18184,6.785,,,
delayed_information,delayed_information,lookahead=2,Exact-lookahead-2,,1000,0.001,-0.015,0.016512499999999933,0.0,,,
delayed_information,delayed_information,lookahead=5,Exact-lookahead-5,,1000,0.17965,0.17711975,0.18187025,6.785,,,
matched_beta,beta_binomial,matched,Count-Bellman-fitted,500.0,1000,0.11313999999999999,0.10447899999999999,0.12155999999999997,4.618,,,
matched_beta,beta_binomial,matched,Count-Bellman-fitted,2000.0,1000,0.11313999999999999,0.10437899999999999,0.12161999999999998,4.618,,,
matched_beta,beta_binomial,matched,Count-Bellman-fitted,50.0,1000,0.11213999999999999,0.10487949999999997,0.11925999999999998,5.543,,,
matched_beta,beta_binomial,matched,Count-Bellman-fitted,100.0,1000,0.11213999999999999,0.10479749999999997,0.11969999999999997,5.543,,,
matched_beta,beta_binomial,matched,Count-Bellman-true,25.0,1000,0.11171999999999999,0.10297849999999999,0.11978049999999998,4.864,,,
matched_beta,beta_binomial,matched,Count-Bellman-true,50.0,1000,0.11171999999999999,0.10332,0.11996099999999998,4.864,,,
matched_beta,beta_binomial,matched,Count-Bellman-true,100.0,1000,0.11171999999999999,0.10294,0.11980149999999998,4.864,,,
matched_beta,beta_binomial,matched,Count-Bellman-true,500.0,1000,0.11171999999999999,0.1029185,0.11959999999999998,4.864,,,
matched_beta,beta_binomial,matched,Count-Bellman-true,2000.0,1000,0.11171999999999999,0.103,0.12006049999999997,4.864,,,
matched_beta,beta_binomial,matched,Amortized-count,500.0,1000,0.11144,0.10325949999999999,0.11968049999999998,5.053,,,
matched_beta,beta_binomial,matched,Amortized-count,2000.0,1000,0.11134,0.1032795,0.11912049999999999,5.058,,,
matched_beta,beta_binomial,matched,Amortized-count,50.0,1000,0.11091999999999999,0.10427949999999998,0.11718049999999998,5.929,,,
matched_beta,beta_binomial,matched,Amortized-count,100.0,1000,0.11055999999999998,0.10443949999999998,0.11626049999999997,6.122,,,
matched_beta,beta_binomial,matched,Amortized-count,25.0,1000,0.10916,0.10386,0.11435999999999999,6.442,,,
matched_beta,beta_binomial,matched,Count-Bellman-fitted,25.0,1000,0.10913999999999999,0.10389949999999999,0.1141,6.443,,,
matched_beta,beta_binomial,matched,Hit-probability-cutoff,25.0,1000,0.08654,0.08465950000000001,0.08862,8.173,,,
matched_beta,beta_binomial,matched,Hit-probability-cutoff,50.0,1000,0.08654,0.08460000000000001,0.08854050000000002,8.173,,,
matched_beta,beta_binomial,matched,Hit-probability-cutoff,100.0,1000,0.08654,0.08452000000000001,0.08848050000000002,8.173,,,
matched_beta,beta_binomial,matched,Hit-probability-cutoff,500.0,1000,0.08654,0.08459950000000001,0.08854000000000001,8.173,,,
matched_beta,beta_binomial,matched,Hit-probability-cutoff,2000.0,1000,0.08654,0.08454,0.08856,8.173,,,
matched_beta,beta_binomial,matched,Fixed-budget-half,25.0,1000,0.07597999999999999,0.06441849999999999,0.08642099999999998,4.951,,,
matched_beta,beta_binomial,matched,Fixed-budget-half,50.0,1000,0.07597999999999999,0.06525949999999998,0.08648049999999999,4.951,,,
matched_beta,beta_binomial,matched,Fixed-budget-half,100.0,1000,0.07597999999999999,0.06453949999999999,0.08670049999999999,4.951,,,
matched_beta,beta_binomial,matched,Fixed-budget-half,500.0,1000,0.07597999999999999,0.064979,0.08704049999999999,4.951,,,
matched_beta,beta_binomial,matched,Fixed-budget-half,2000.0,1000,0.07597999999999999,0.06501799999999999,0.08729999999999999,4.951,,,
matched_beta,beta_binomial,matched,Zero-screen,50.0,1000,0.071,0.056,0.085,0.0,,,
matched_beta,beta_binomial,matched,Zero-screen,100.0,1000,0.071,0.0565,0.0855,0.0,,,
matched_beta,beta_binomial,matched,Zero-screen,500.0,1000,0.071,0.0565,0.085,0.0,,,
matched_beta,beta_binomial,matched,Zero-screen,2000.0,1000,0.071,0.0555,0.086,0.0,,,
matched_beta,beta_binomial,matched,Full-screen,25.0,1000,0.05,0.05,0.05,10.0,,,
matched_beta,beta_binomial,matched,Full-screen,50.0,1000,0.05,0.05,0.05,10.0,,,
matched_beta,beta_binomial,matched,Full-screen,100.0,1000,0.05,0.05,0.05,10.0,,,
matched_beta,beta_binomial,matched,Full-screen,500.0,1000,0.05,0.05,0.05,10.0,,,
matched_beta,beta_binomial,matched,Full-screen,2000.0,1000,0.05,0.05,0.05,10.0,,,
matched_beta,beta_binomial,matched,Zero-screen,25.0,1000,-0.071,-0.086,-0.056,0.0,,,
mixture_shift,mixture_shift,shifted_test,Rank-score-latent-full,,1000,0.19884000000000002,0.1899365,0.20748049999999998,0.308,,,
mixture_shift,mixture_shift,shifted_test,Count-Bellman-fitted,,1000,0.1945,0.1844875,0.204,0.0,,,
mixture_shift,mixture_shift,shifted_test,Full-screen,,1000,0.05,0.05,0.05,10.0,,,
perfect_information,perfect_information,cost=0.01,Bayes-exact,,1000,0.2400000000000001,0.2400000000000001,0.2400000000000001,1.0,,,
rank_decay,rank_decay,gamma=0.0,True-rank-oracle,,1000,0.24506,0.24245950000000002,0.2473005,0.122,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.0,Count-Bellman-fitted,,1000,0.244,0.2405,0.247,0.0,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.0,Rank-only-full,,1000,0.244,0.2405,0.247,0.0,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.0,Rank-score-local-full,,1000,0.244,0.2405,0.247,0.0,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.0,Rank-score-mean-full,,1000,0.244,0.2405,0.247,0.0,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.0,Rank-score-real-full,,1000,0.244,0.2405,0.247,0.0,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.0,Rank-score-shuffled-across-full,,1000,0.244,0.2404875,0.247,0.0,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.0,Rank-score-shuffled-within-full,,1000,0.244,0.2405,0.247,0.0,0.02,0.24211747642802553,0.0009604480081575017
rank_decay,rank_decay,gamma=0.5,True-rank-oracle,,1000,0.2266,0.22047999999999998,0.23188,0.495,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=0.5,Rank-score-mean-full,,1000,0.22348,0.2168195,0.2298405,0.251,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=0.5,Rank-score-real-full,,1000,0.22348,0.2167995,0.229682,0.251,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=0.5,Rank-score-shuffled-within-full,,1000,0.22314,0.21606,0.22918049999999998,0.243,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=0.5,Count-Bellman-fitted,,1000,0.2175,0.21,0.22451249999999995,0.0,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=0.5,Rank-only-full,,1000,0.2175,0.2095,0.2245,0.0,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=0.5,Rank-score-local-full,,1000,0.2175,0.2095,0.225,0.0,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=0.5,Rank-score-shuffled-across-full,,1000,0.2175,0.21,0.2245,0.0,0.092,0.22211256298828272,0.0033446022697264234
rank_decay,rank_decay,gamma=1.5,True-rank-oracle,,1000,0.17586,0.16833949999999998,0.18309999999999996,2.532,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=1.5,Rank-score-mean-full,,1000,0.15827999999999998,0.15215949999999998,0.16397999999999996,3.961,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=1.5,Rank-score-shuffled-within-full,,1000,0.15824,0.1525,0.16423999999999997,3.963,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=1.5,Rank-score-real-full,,1000,0.15775999999999998,0.15194,0.16326049999999998,4.037,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=1.5,Rank-only-full,,1000,0.14855999999999997,0.14425949999999998,0.15260049999999994,4.897,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=1.5,Rank-score-local-full,,1000,0.14765999999999996,0.14291949999999998,0.15215999999999996,4.842,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=1.5,Rank-score-shuffled-across-full,,1000,0.14749999999999996,0.14154,0.15320099999999995,4.475,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=1.5,Count-Bellman-fitted,,1000,0.10947999999999998,0.09849899999999999,0.1196,3.401,0.589,0.13667784290745907,0.03203369969980589
rank_decay,rank_decay,gamma=3.0,True-rank-oracle,,1000,0.21869999999999998,0.21276,0.22435999999999998,0.79,0.331,0.19853595581634145,0.014860993853763193
rank_decay,rank_decay,gamma=3.0,Rank-score-mean-full,,1000,0.21559999999999999,0.2100195,0.2211805,0.995,0.331,0.19853595581634145,0.014860993853763193
rank_decay,rank_decay,gamma=3.0,Rank-score-shuffled-within-full,,1000,0.21459999999999996,0.20851999999999998,0.22018200000000002,0.995,0.331,0.19853595581634145,0.014860993853763193
rank_decay,rank_decay,gamma=3.0,Rank-score-real-full,,1000,0.21427999999999997,0.20827899999999996,0.22008049999999998,1.036,0.331,0.19853595581634145,0.014860993853763193
rank_decay,rank_decay,gamma=3.0,Rank-score-shuffled-across-full,,1000,0.2037,0.19723849999999998,0.209981,1.24,0.331,0.19853595581634145,0.014860993853763193
rank_decay,rank_decay,gamma=3.0,Rank-only-full,,1000,0.20316000000000004,0.197959,0.20800149999999998,1.767,0.331,0.19853595581634145,0.014860993853763193
rank_decay,rank_decay,gamma=3.0,Rank-score-local-full,,1000,0.20316000000000004,0.1978,0.208141,1.767,0.331,0.19853595581634145,0.014860993853763193
rank_decay,rank_decay,gamma=3.0,Count-Bellman-fitted,,1000,0.203,0.194,0.2115,0.0,0.331,0.19853595581634145,0.014860993853763193

## Planner summary

method,states,high_gap_action_agreement,mean_planning_regret,maximum_planning_regret
Exact-lookahead-1,152,0.6842105263157895,0.016387259800086127,0.08251014492753617
Exact-lookahead-10,152,1.0,0.0,0.0
Exact-lookahead-2,152,0.8486842105263158,0.003458861725211234,0.02500676328502416
Exact-lookahead-5,152,1.0,0.0,0.0

## Recall summary

suite,environment,condition,method,target_recall,alpha,claims_safety,topics,failure_rate,failure_ci_low,failure_ci_high,failure_ucb_95,reliability_gap,mean_review_fraction,mean_excess_effort,mean_overshoot
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.8,0.05,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.796407,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.8,0.1,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.796407,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.9,0.05,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.8958455,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.9,0.1,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.8958455,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.95,0.05,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.9453744999999999,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Recall-oracle,0.95,0.1,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.9453744999999999,0.0,0.0
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.8,0.1,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.9646394999999999,33.6465,0.16470599999999994
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.8,0.05,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.9648700000000001,33.6926,0.16478399999999993
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Monotone-rank-plug-in-UCB,0.95,0.05,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.9700764999999999,4.9404,0.020184000000000018
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Monotone-rank-plug-in-UCB,0.95,0.1,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.9700764999999999,4.9404,0.020184000000000018
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.9,0.1,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.9709615000000001,15.0232,0.07117799999999995
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.9,0.05,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.971245,15.0799,0.07138999999999995
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.95,0.1,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.9731079999999999,5.5467,0.02313200000000002
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.95,0.05,False,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.9780785,6.5408,0.028062000000000024
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.95,0.05,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,0.9933965,9.6044,0.04339200000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Sequential-hypergeometric,0.95,0.1,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,0.9934755000000001,9.6202,0.043459000000000046
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.8,0.05,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,1.0,40.7186,0.19999999999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.8,0.1,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,1.0,40.7186,0.19999999999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.9,0.05,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,1.0,20.8309,0.09999999999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.9,0.1,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,1.0,20.8309,0.09999999999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.95,0.05,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.05,1.0,10.9251,0.050000000000000044
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Full-screen,0.95,0.1,True,10000,0.0,0.0,0.0,0.00029952835977661195,-0.1,1.0,10.9251,0.050000000000000044
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.8,0.05,False,10000,0.0001,0.0,0.0003,0.0004742976591654368,-0.0499,0.9999845000000002,40.7155,0.19997799999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.8,0.1,False,10000,0.0001,0.0,0.0003,0.0004742976591654368,-0.0999,0.9999845000000002,40.7155,0.19997799999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.9,0.05,False,10000,0.0001,0.0,0.0003,0.0004742976591654368,-0.0499,0.9999845000000002,20.8278,0.09997799999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.9,0.1,False,10000,0.0001,0.0,0.0003,0.0004742976591654368,-0.0999,0.9999845000000002,20.8278,0.09997799999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.95,0.05,False,10000,0.0001,0.0,0.0003,0.0004742976591654368,-0.0499,0.9999845000000002,10.922,0.04997800000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Consecutive-miss-20,0.95,0.1,False,10000,0.0001,0.0,0.0003,0.0004742976591654368,-0.0999,0.9999845000000002,10.922,0.04997800000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Monotone-rank-plug-in-UCB,0.9,0.05,False,10000,0.0022,0.0014,0.0031,0.0031400019060426844,-0.0478,0.930168,6.8645,0.030121999999999986
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.95,0.05,True,10000,0.0023,0.0014,0.0033,0.003256976390880396,-0.047700000000000006,0.9934689999999998,9.6189,0.04351900000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.95,0.1,True,10000,0.0034,0.0023,0.0046,0.004524010463635399,-0.0966,0.9933774999999999,9.6006,0.04344300000000004
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.9,0.05,False,10000,0.004,0.0028,0.0054,0.0052037929808234195,-0.046,0.9394020000000001,8.7113,0.03896599999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.8,0.1,True,10000,0.0061,0.0046,0.0077,0.007543994794539046,-0.09390000000000001,0.9327235,27.2633,0.13278299999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.8,0.05,True,10000,0.0077,0.006,0.0094,0.009299797961032797,-0.042300000000000004,0.9318445,27.0875,0.13172799999999996
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Monotone-rank-plug-in-UCB,0.9,0.1,False,10000,0.0089,0.0071,0.0108,0.010606077737945712,-0.0911,0.9253849999999999,5.9079,0.02539799999999999
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Monotone-rank-plug-in-UCB,0.8,0.05,False,10000,0.0096,0.0077,0.0116,0.011364710187575237,-0.040400000000000005,0.8396384999999998,8.6463,0.040075999999999945
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.9,0.1,False,10000,0.0114,0.0094,0.0135,0.013306099034683638,-0.08860000000000001,0.931334,7.0977,0.03113399999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.9,0.05,True,10000,0.0155,0.0132,0.0179,0.017690056955550534,-0.0345,0.9774164999999999,16.3142,0.07748499999999997
ideal_random_order,sparse_recall,"r=100,gamma=0.0",QBCB,0.9,0.1,True,10000,0.0155,0.013097500000000002,0.018,0.017690056955550534,-0.0845,0.9774305,16.317,0.07738099999999998
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Monotone-rank-plug-in-UCB,0.8,0.1,False,10000,0.0256,0.0225,0.0288,0.028352659973185264,-0.07440000000000001,0.8315869999999999,7.036,0.03210299999999994
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.8,0.05,False,10000,0.0425,0.0385,0.04660249999999999,0.04596830167444219,-0.0075,0.846842,10.087,0.04658699999999994
ideal_random_order,sparse_recall,"r=100,gamma=0.0",Point-process-exponential,0.8,0.1,False,10000,0.0819,0.0765,0.0871,0.0865510998424781,-0.018100000000000005,0.8358774999999999,7.8941,0.03550699999999995
sparse_recall,sparse_recall,"r=5,gamma=2.0",Recall-oracle,0.9,0.05,False,2000,0.0,0.0,0.0,0.0014967448951882837,-0.05,0.6742149999999999,0.0,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Full-screen,0.9,0.05,True,2000,0.0,0.0,0.0,0.0014967448951882837,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Monotone-rank-plug-in-UCB,0.9,0.05,False,2000,0.0,0.0,0.0,0.0014967448951882837,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Point-process-exponential,0.9,0.05,False,2000,0.0,0.0,0.0,0.0014967448951882837,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",QBCB,0.9,0.05,True,2000,0.0,0.0,0.0,0.0014967448951882837,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Sequential-hypergeometric,0.9,0.05,True,2000,0.0,0.0,0.0,0.0014967448951882837,-0.05,1.0,162.8925,0.1
sparse_recall,sparse_recall,"r=5,gamma=2.0",Consecutive-miss-20,0.9,0.05,False,2000,1.0,1.0,1.0,1.0,0.95,0.04992,-312.1475,-0.7985

## Gates

```json
{
  "A_oracle_correctness": {
    "perfect_information_test": "pytest test_hindsight_flow_target_is_not_bayes_optimal",
    "status": "pass"
  },
  "B_matched_model": {
    "status": "reported",
    "true_minus_fitted_utility_by_training_topics": {
      "25": 0.002579999999999999,
      "50": -0.0004199999999999898,
      "100": -0.0004199999999999898,
      "500": -0.0014200000000000046,
      "2000": -0.0014200000000000046
    },
    "note": "finite held-out differences are reported; true model is the exact oracle in the unit gate"
  },
  "C_rank_aware": {
    "status": "pass",
    "gamma_1_5_mean_utility": {
      "Count-Bellman-fitted": 0.10948000000000001,
      "Rank-only-full": 0.14856,
      "Rank-score-local-full": 0.14765999999999999,
      "Rank-score-mean-full": 0.15828,
      "Rank-score-real-full": 0.15775999999999998,
      "Rank-score-shuffled-across-full": 0.1475,
      "Rank-score-shuffled-within-full": 0.15824000000000002,
      "True-rank-oracle": 0.17586
    },
    "true_oracle_regret_reduction_vs_count": 0.7273275082856281,
    "next_hit_calibration": {
      "Rank-only": {
        "brier": 0.15286407324625695,
        "log_loss": 0.4719767299396135,
        "ece": 0.03198634731278307
      },
      "Rank-score-mean": {
        "brier": 0.14425656257459754,
        "log_loss": 0.4510434506999951,
        "ece": 0.04568136439233675
      },
      "Rank-score-local": {
        "brier": 0.15242949955069537,
        "log_loss": 0.4701423458669895,
        "ece": 0.029411339006937078
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
    "status": "pass",
    "full_lookahead_high_gap_agreement": 1.0,
    "rank_exact_test": "pytest test_rank_belief_full_horizon_matches_exact_oracle"
  },
  "E_recall": {
    "status": "pass",
    "one_sided_confidence": 0.95,
    "tolerance": 0.01,
    "worst_cell_by_method": {
      "Full-screen": {
        "worst_cell_ucb_95": 0.0014967448951882837,
        "worst_cell_ucb_excess": -0.04850325510481172
      },
      "QBCB": {
        "worst_cell_ucb_95": 0.017690056955550534,
        "worst_cell_ucb_excess": -0.032309943044449466
      },
      "Sequential-hypergeometric": {
        "worst_cell_ucb_95": 0.0014967448951882837,
        "worst_cell_ucb_excess": -0.04850325510481172
      }
    }
  }
}
```
