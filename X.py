import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load dataset
data = pd.read_csv("financial_data_100k.csv")

# Select features and target
features = ["P/E Ratio", "P/B Ratio", "D/E Ratio", "ROE", "ROA"]
target = "High_Performance"

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save if accuracy ≥ 95%
if accuracy >= 0.95:
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✅ Model and scaler saved.")
else:
    print("❌ Accuracy below 95%. Model not saved.")
