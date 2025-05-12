from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features
    features = [float(request.form[feature]) for feature in [
        'worst perimeter', 'mean concave points', 'worst radius',
        'mean perimeter', 'worst concave points'
    ]]
    # Scale features
    features_scaled = scaler.transform([features])
    # Predict
    prediction = model.predict(features_scaled)[0]
    result = 'Malignant' if prediction == 0 else 'Benign'
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)