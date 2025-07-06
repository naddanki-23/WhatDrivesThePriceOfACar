# WhatDrivesThePriceOfACar
Model Performance Summary (with Cross-Validation)

The table and chart below summarize the performance of three regression models—Linear Regression, Ridge Regression, and Lasso Regression—evaluated on a dataset of used car listings.

Model

CV R² (mean ± std)

Test R²

Test RMSE (log)

Test MAE (log)

Best Hyperparameters

Linear Regression

0.843943 ± 0.013578

0.840468

0.324232

0.225463

None

Ridge Regression

0.844739 ± 0.013118

0.841528

0.323153

0.225619

{'regressor__alpha': 1}

Lasso Regression

0.838079 ± 0.010577

0.840082

0.324625

0.228625

{'regressor__alpha': 0.001}

Key Observations:

All three models exhibit similar performance, with Ridge Regression achieving the best test R² (0.8415) and lowest RMSE.

Linear Regression performs nearly as well, while Lasso shows slightly higher error but enables feature selection.

These results were derived using 5-fold cross-validation and a 20% hold-out test set.

The accompanying bar plot visualizes the cross-validated R² scores with error bars (standard deviation), highlighting the consistency of model performance across folds.

Visual Interpretation:

The bar plot comparing models with error bars helps visualize model stability: Ridge and Linear Regression show nearly identical mean R² values, with Ridge having slightly narrower error margins.

This suggests Ridge may generalize slightly better across unseen data.

Lasso, while slightly lower in R², is valuable when simpler models or feature selection is desired.

These visual insights complement the numerical results and guide the choice of model depending on the application’s priority: accuracy vs. interpretability.

