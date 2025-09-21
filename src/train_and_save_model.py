# src/train_and_save_model.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
DATA_PATH = "data/flood_data.csv"
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Preprocessing
# ---------------------------
# Encode categorical features
for col in ["Land Cover", "Soil Type"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
feature_cols = [c for c in df.columns if c != "Flood Occurred"]
X = df[feature_cols]
y = df["Flood Occurred"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/flood_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nSaved model -> models/flood_model.pkl")
print("Saved scaler -> models/scaler.pkl")
