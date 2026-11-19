# Life Expectancy Prediction

Predicting national life expectancy using World Health Organization (WHO) health, immunization, and economic indicators (2000–2015). Because the raw dataset contains missing values, skewed distributions, and outliers, this project focuses on thorough data cleaning and feature engineering before comparing regression models.

## Dataset
- **Source:** [Life Expectancy (WHO) on Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **Size:** 2,938 rows, 22 columns (2,928 rows retained after cleaning)

## Data Cleaning & Feature Engineering
- **Column Normalization:** Stripped hidden leading, trailing, and duplicate whitespace from column names.
- **Outlier Handling:** Replaced impossible `BMI` values (below 10 or above 70) with `NaN` rather than dropping rows, preserving 99.6% of the dataset.
- **Imputation:** Filled missing numerical values using each country's median first, falling back to the global median to avoid distorting country-level trends.
- **Feature Engineering:**
  - Encoded `Status` (`Developed` / `Developing`) into a binary feature.
  - Created an `Avg_Immunization` composite score from `Hepatitis B`, `Polio`, and `Diphtheria`.
  - Applied `log1p` transformations to heavily right-skewed features (`GDP`, `Population`, `HIV/AIDS`, `Measles`).
  - Dropped multicollinear columns (`under-five deaths`, `thinness 5-9 years`).

## Model Results

| Model | MAE (Years) | R² Score |
|---|---|---|
| Linear Regression (Baseline) | 2.77 | 0.848 |
| Random Forest Regressor | **1.07** | **0.965** |

## Project Structure

```text
life-expectancy-prediction/
├── data/
│   ├── raw/                  # Original WHO CSV
│   └── processed/            # Cleaned & engineered dataset
├── notebooks/                # Exploratory Data Analysis (EDA)
├── src/
│   ├── clean_data.py         # Data cleaning & feature engineering pipeline
│   └── train_model.py        # Model training & evaluation
├── requirements.txt
└── README.md
