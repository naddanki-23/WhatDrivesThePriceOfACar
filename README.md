# Module 11 Practical Application Assignment - What drives the price of a car?

## Overview

The goal of this project is to build regression models that predict the log-transformed price of used vehicles based on their characteristics (year, make, mileage, etc.) and to analyze/ identify the factors that influence a car price.
Dataset
Source: Craigslist vehicle listings dataset
Size: 42,000+ rows before cleaning, ~31,000 after cleaning
Target Variable: price_log — the natural log of the vehicle's listed price

## Notebooks
- coupon_analysis.ipynb: [link to colab notebook code]([https://drive.google.com/file/d/1FaYJIwel29aY514VXwAM2iGC_nz1itWV/view?usp=sharing](https://drive.google.com/file/d/1j8aCwJ5I4aJp9V_eHGrjQLHVO2qX6LuH/view?usp=sharing))

## Data Description 
- id – Unique listing ID
- price – Listing price of the vehicle (target variable)
- year – Manufacturing year of the vehicle
- manufacturer – Manufacturer/brand of the vehicle (e.g., Ford, Toyota)
- model – Specific model of the vehicle (e.g., Camry, Civic)
- condition – Condition of the vehicle (e.g., new, like new, excellent, good, fair, salvage)
- cylinders – Number of engine cylinders (e.g., 4 cylinders, 6 cylinders)
- fuel – Type of fuel used (e.g., gas, diesel, electric, hybrid)
- odometer – Mileage of the vehicle (in miles)
- title_status – Legal ownership status (e.g., clean, rebuilt, salvage)
- transmission – Transmission type (e.g., automatic, manual)
- VIN – Vehicle Identification Number
- drive – Drive type (e.g., 4wd, fwd, rwd)
- size – Size category of the vehicle (e.g., compact, full-size)
- type – Vehicle body type (e.g., SUV, truck, sedan)
- paint_color – Exterior color
- state – U.S. state where the vehicle is listed

## Data Cleaning/Preprocessing
### Data Cleaning 
- Filtered out outliers based on price, odometer, and year using IQR
- *Drop 4 Columns due to missing values/High-cardinality**:
  * `region` , `VIN` both had High-cardinalityand were dropped
  * `model` and `id` had too many unique values.

### Feature Engineering
- Created new feature: age = 2025 - year
   - Features that ended up beign used Used: year, odometer, manufacturer, condition, cylinders, fuel, title_status, transmission, drive, type, paint_color, state, area, "age"
- Applied log transformation to price to handle skewness
- Handled categorical variables using one-hot encoding
  
### EDA and Correlation Analysis
- * Perform univariate and bivariate analysis to:
  * Understand distributions (histograms)
  * Identify correlations with target (`price`) using heatmaps 

    <img src="images/ corr_mat_price_log.png" width="600"/>

- Scaled numerical features using standardization
- Added polynomial features (degree=2) to capture interaction and non-linear effects among numeric predictors



---


---



---



---


- 
## Model Performance Summary (with Cross-Validation)

<img src="images/table.png" width="850"/>
### Key Observations:

All three models exhibit similar performance, with Ridge Regression achieving the best test R² (0.8415) and lowest RMSE.
Linear Regression performs nearly as well, while Lasso shows slightly higher error but enables feature selection.
These results were derived using 5-fold cross-validation and a 20% hold-out test set.
The accompanying bar plot visualizes the cross-validated R² scores with error bars (standard deviation), highlighting the consistency of model performance across folds.

### Visual Interpretation:

The bar plot comparing models with error bars helps visualize model stability: Ridge and Linear Regression show nearly identical mean R² values, with Ridge having slightly narrower error margins.
This suggests Ridge may generalize slightly better across unseen data.
Lasso, while slightly lower in R², is valuable when simpler models or feature selection is desired.
These visual insights complement the numerical results and guide the choice of model depending on the application’s priority: accuracy vs. interpretability.

## Key Features Impacting Used Car Price

Based on the coefficient analysis from the best-performing model (Ridge or Lasso), the following features had the strongest impact on price predictions:

### Top Predictive Features effect on price: 
- odometer: Higher mileage lowers vehicle value.
- year / age: Newer vehicles tend to be priced higher.
- fuel_diesel:Diesel-powered vehicles have higher resale value.
- condition_like new: Excellent condition significantly boosts price.
- drive_4wd:All-wheel/4WD vehicles typically command higher prices.
- title_status_clean: Clean title increases trust and resale value.
- transmission_automatic: Automatics are generally more desirable.

### Low-Impact Features:
Features such as paint_color, cylinders, and certain state or type values showed very low or zero importance (especially in Lasso), indicating minimal predictive contribution.
These insights help prioritize which attributes matter most in pricing models and inform future data collection or simplification efforts.

## Findings

### Business Understanding
- What are the most important factors that determine used car prices, and can we build a reliable model to help with inventory decisions?
- Our goal was to deliver clear, actionable insights to help you price, buy, and sell smarter.

## Key Findings & Actionable Items

- **Year:** Newer vehicles sell for significantly more.
- **Odometer:** Lower mileage = higher price. Mileage is one of the top predictors.
- **Condition:** “New” and “excellent” cars earn the best prices. 

**What should you do?**
- **Sell smart:** Highlight the best features (condition, brand) in your ads.
- **Price smart:** Set realistic prices for high-mileage or salvage vehicles to move them quickly.
- **Buy smart:** Focus on newer, low-mileage, well-maintained cars.

## Next Steps & Recommendations
- Add more features such as accident history ot try advanced model for stronger predictions
- Build a web app using Streamlit or Flask for real-time predictions
- If we had more data we could go futhers and break the results by region or time 




