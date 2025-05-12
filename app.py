from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load logistic regression model and scaler
try:
    model = joblib.load('log_reg_model.pkl')  # Renamed from regression_model_7.pkl
    scaler = joblib.load('scaler.pkl')  # Renamed from regression_scaler_7.pkl
except FileNotFoundError:
    print("Error: Model or scaler file not found. Ensure log_reg_model.pkl and scaler.pkl are in the web_app directory.")
    exit(1)

# List of 7 selected features (match data.feature_names)
features = [
    'radius error',
    'worst concave points',
    'area error',
    'compactness error',
    'perimeter error',
    'worst texture',
    'fractal dimension error'
]


@app.route('/')
def home():
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features
        feature_values = []
        for feature in features:
            value = float(request.form[feature])
            if value < 0:
                return render_template('result.html', prediction="Error: All measurements must be non-negative.")
            feature_values.append(value)

        # Scale features
        features_scaled = scaler.transform([feature_values])

        # Predict (logistic regression outputs probability, threshold at 0.5)
        prediction_proba = model.predict_proba(features_scaled)[0]
        prediction = 0 if prediction_proba[0] > 0.5 else 1  # Class 0 (Malignant) or 1 (Benign)
        result = 'Malignant' if prediction == 0 else 'Benign'
        return render_template('result.html', prediction=result)
    except ValueError:
        return render_template('result.html', prediction="Error: Please enter valid numeric values for all fields.")
    except Exception as e:
        return render_template('result.html', prediction=f"Error: An unexpected issue occurred ({str(e)}).")


if __name__ == '__main__':
    app.run(debug=True)