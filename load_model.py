import pickle
import numpy as np

# Load the trained XGBoost model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the StandardScaler used during training
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Example usage with dummy raw (unscaled) features
# Format: [P/E Ratio, P/B Ratio, D/E Ratio, ROE, ROA]
sample_features = np.array([[10.5, 1.2, 0.8, 12.3, 8.5]])

# Scale features before prediction
sample_scaled = scaler.transform(sample_features)

# Make prediction
prediction = model.predict(sample_scaled)
prediction_proba = model.predict_proba(sample_scaled)

# Show results
print("Sample Prediction:", "Buy" if prediction[0] == 1 else "Do not buy")
print("Confidence Score:", prediction_proba[0][1] * 100 if prediction[0] == 1 else prediction_proba[0][0] * 100, "%")
