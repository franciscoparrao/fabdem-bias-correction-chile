# T2 — XGBoost hyperparameters

| Parameter               | Value   | Description                                       |
|:------------------------|:--------|:--------------------------------------------------|
| n\_estimators           | 1000    | Trees in ensemble                                 |
| max\_depth              | 9       | Maximum tree depth                                |
| learning\_rate (eta)    | 0.0113  | Shrinkage rate (low $\to$ more regularization)    |
| subsample               | 0.953   | Row subsample ratio per tree                      |
| colsample\_bytree       | 0.635   | Feature subsample ratio per tree                  |
| min\_child\_weight      | 14      | Min child node weight (regularization)            |
| reg\_lambda (L2)        | 7.410   | L2 weight regularization                          |
| reg\_alpha (L1)         | 1.597   | L1 weight regularization                          |
| gamma                   | 2.202   | Min loss reduction for split (post-pruning)       |
| early\_stopping\_rounds | 30      | Stops if eval RMSE stops improving                |
| tree\_method            | hist    | Histogram-based split finding                     |
| Optuna trials           | 100     | TPE sampler, objective = mean spatial-CV OOF RMSE |
| random\_state           | 42      |                                                   |
