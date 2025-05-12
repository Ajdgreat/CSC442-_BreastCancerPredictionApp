import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load Breast Cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Select top 5 features
features = ['worst perimeter', 'mean concave points', 'worst radius',
            'mean perimeter', 'worst concave points']
X = df[features]
y = data.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved as rf_model.pkl and scaler.pkl")

# Test loading
loaded_model = joblib.load('rf_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
print("Model and scaler loaded successfully")