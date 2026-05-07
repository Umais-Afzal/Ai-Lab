import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# --------------------------------------------------------------
# 1. Load dataset
# --------------------------------------------------------------
df = pd.read_csv("data.csv")
print("Original shape:", df.shape)

# --------------------------------------------------------------
# 2. Preprocessing – handle missing values safely
# --------------------------------------------------------------
# Numeric columns: fill NaN with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Object (string) columns: fill NaN with mode (most frequent)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val[0])
    else:
        df[col] = df[col].fillna("Unknown")

# Convert float columns to Int64 (nullable integer) – but do it after filling NaNs
float_cols = df.select_dtypes(include=['float64']).columns
for col in float_cols:
    df[col] = df[col].round().astype('Int64')

# Convert 'date' to datetime (will be dropped later)
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# --------------------------------------------------------------
# 3. Encode categorical columns (text → numbers)
# --------------------------------------------------------------
cat_cols = df.select_dtypes(include=['object']).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
print("Categorical columns encoded:", list(cat_cols))

# --------------------------------------------------------------
# 4. Define features (X) and target (y)
# --------------------------------------------------------------
target = 'price'
X = df.drop(columns=[target, 'date'])   # drop 'date' – it's not a good predictor
y = df[target]

# Ensure y has no NaN (drop rows where price is NaN)
if y.isnull().any():
    print(f"Dropping {y.isnull().sum()} rows where target is NaN")
    valid = ~y.isnull()
    X = X[valid]
    y = y[valid]

# --------------------------------------------------------------
# 5. Final safety: convert all X columns to numeric and fill any leftover NaN with 0
# --------------------------------------------------------------
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

print("Features shape after cleaning:", X.shape)
print("Any NaN in X?", X.isnull().any().any())
print("Any NaN in y?", y.isnull().any())

# --------------------------------------------------------------
# 6. Train‑test split
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------
# 7. Train Linear Regression model
# --------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
r2 = model.score(X_test, y_test)
print(f"Model trained. R² score on test set: {r2:.4f}")

# --------------------------------------------------------------
# 8. Save model and encoders for later use in Flask
# --------------------------------------------------------------
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')
print("Model and encoders saved.")