📘 Project Overview: Used Car Price Prediction

🧾 Objective

The goal of this project is to build regression models that predict the log-transformed price of used vehicles based on their characteristics (year, make, mileage, etc.). We evaluate and compare several models using cross-validation and test performance metrics.

📊 Dataset

Source: Craigslist vehicle listings dataset

Size: 42,000+ rows before cleaning, ~31,000 after cleaning

Target Variable: price_log — the natural log of the vehicle's listed price

Features Used: year, odometer, manufacturer, condition, cylinders, fuel, title_status, transmission, drive, type, paint_color, state

🧹 Data Cleaning and Preprocessing

Removed rows with missing values (threshold: ≤4 missing columns)

Filtered out outliers based on price, odometer, and year using IQR

Created new feature: age = 2025 - year

Applied log transformation to price to handle skewness

Handled categorical variables using one-hot encoding

Scaled numerical features using standardization

Added polynomial features (degree=2) to capture interaction and non-linear effects among numeric predictors

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

🔍 Key Features Impacting Used Car Price

Based on the coefficient analysis from the best-performing model (Ridge or Lasso), the following features had the strongest impact on price predictions:

📌 Top Predictive Features:

Feature

Effect on Price

Interpretation

odometer

🔻 Negative

Higher mileage lowers vehicle value.

year / age

🔺 Positive

Newer vehicles tend to be priced higher.

fuel_diesel

🔺 Positive

Diesel-powered vehicles have higher resale value.

condition_like new

🔺 Positive

Excellent condition significantly boosts price.

drive_4wd

🔺 Positive

All-wheel/4WD vehicles typically command higher prices.

title_status_clean

🔺 Positive

Clean title increases trust and resale value.

transmission_automatic

🔺 Positive

Automatics are generally more desirable.

❗ Low-Impact Features:

Features such as paint_color, cylinders, and certain state or type values showed very low or zero importance (especially in Lasso), indicating minimal predictive contribution.

These insights help prioritize which attributes matter most in pricing models and inform future data collection or simplification efforts.

📌 Next Steps (Optional Ideas)

Add more advanced polynomial features or interaction terms

Use ensemble methods (e.g., Random Forests, XGBoost) for comparison

Incorporate real dollar value predictions (by reversing the log)

Build a web app using Streamlit or Flask for real-time predictions
