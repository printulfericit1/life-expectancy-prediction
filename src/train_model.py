import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_life_expectancy.csv")

X = df.drop(columns=["Life expectancy"])
y = df["Life expectancy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for Linear Regression (fit on train only to prevent data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Baseline: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
print(
    f"Linear Regression -> MAE: {mean_absolute_error(y_test, lr_preds):.2f} years | "
    f"R2: {r2_score(y_test, lr_preds):.3f}"
)

# 2. Tree-based: Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(
    f"Random Forest     -> MAE: {mean_absolute_error(y_test, rf_preds):.2f} years | "
    f"R2: {r2_score(y_test, rf_preds):.3f}"
)