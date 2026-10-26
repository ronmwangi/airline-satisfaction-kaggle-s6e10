# Airline Passenger Satisfaction: Kaggle Playground S6E10

Binary classification: predict the probability that an airline passenger is satisfied. Metric: ROC AUC.

**Result:** public leaderboard **0.95849** (blend of 5 gradient-boosted models, 5-fold CV).

## Approach
- Categorical handling with consistent categories across train and test
- 5-fold stratified CV; early stopping on each fold
- Out-of-fold (OOF) predictions used to compare every experiment fairly
- Final model: rank-average blend of LightGBM and XGBoost variants

## Experiments (OOF AUC)
| Experiment | OOF AUC |
|---|---|
| LightGBM baseline | 0.95894 |
| + original dataset added to training | 0.95867 |
| + engineered features (rating stats, delay features) | 0.95873 |
| Regularized LightGBM | 0.95887 |
| Blend of 4 LightGBM models | 0.95920 |
| XGBoost | 0.95890 |
| Blend of 5 models (final) | 0.95924 |

## Findings
- Adding the original dataset and engineered features did not improve CV, so I dropped them rather than keep complexity that didn't pay off.
- All approaches converged within about 0.0005 AUC of each other, which suggests the data's noise ceiling is near 0.96. Blending diverse models gave the only consistent gain.
- Public leaderboard (0.95849) tracked CV (about 0.959) closely, so the validation setup was reliable.

## Stack
Python, pandas, LightGBM, XGBoost, scikit-learn
