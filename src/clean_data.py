import pandas as pd
import numpy as np


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # 1. Strip hidden whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Drop rows missing the target variable
    df = df.dropna(subset=["Life expectancy"])

    # 3. Filter out unrealistic BMI outliers
    df.loc[(df["BMI"] < 10) | (df["BMI"] > 70), "BMI"] = np.nan

    # 4. Impute missing numerical values (country median -> global median)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df.groupby("Country")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary encoding for Status
    df["Status_Developed"] = (df["Status"] == "Developed").astype(int)

    # Average immunization score
    df["Avg_Immunization"] = df[["Hepatitis B", "Polio", "Diphtheria"]].mean(axis=1)

    # Log-transform heavily skewed columns
    for col in ["GDP", "Population", "HIV/AIDS", "Measles"]:
        df[f"Log_{col}"] = np.log1p(df[col])

    # Drop redundant / raw skewed / non-numeric columns
    drop_cols = [
        "Country",
        "Status",
        "under-five deaths",
        "thinness 5-9 years",
        "GDP",
        "Population",
        "HIV/AIDS",
        "Measles",
    ]
    df = df.drop(columns=drop_cols)

    return df


if __name__ == "__main__":
    raw_path = "data/raw/Life Expectancy Data.csv"
    clean_df = load_and_clean(raw_path)
    final_df = engineer_features(clean_df)

    # Save cleaned dataset
    final_df.to_csv("data/processed/cleaned_life_expectancy.csv", index=False)
    print(f"Done! Saved cleaned dataset with shape: {final_df.shape}")