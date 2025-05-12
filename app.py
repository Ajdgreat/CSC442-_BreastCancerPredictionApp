from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load SVM model and scaler
try:
    model = joblib.load('log_reg_model.pkl')  # Will be replaced with svm_model.pkl
    scaler = joblib.load('scaler.pkl')  # Will be replaced with svm_scaler.pkl
except FileNotFoundError:
    print("Error: Model or scaler file not found. Ensure log_reg_model.pkl and scaler.pkl are in the web_app directory.")
    exit(1)

# Feature lists for sections
mean_features = [
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
    'mean_smoothness', 'mean_compactness', 'mean_concavity',
    'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension'
]
error_features = [
    'radius_error', 'texture_error', 'perimeter_error', 'area_error',
    'smoothness_error', 'compactness_error', 'concavity_error',
    'concave_points_error', 'symmetry_error', 'fractal_dimension_error'
]
worst_features = [
    'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
    'worst_smoothness', 'worst_compactness', 'worst_concavity',
    'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
]
all_features = mean_features + error_features + worst_features


@app.route('/')
def home():
    return render_template('index.html',
                           mean_features=mean_features,
                           error_features=error_features,
                           worst_features=worst_features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features
        feature_values = []
        for feature in all_features:
            value = float(request.form[feature])
            if value < 0:
                return render_template('result.html', prediction="Error: All measurements must be non-negative.")
            feature_values.append(value)

        # Scale features
        features_scaled = scaler.transform([feature_values])

        # Predict
        prediction = model.predict(features_scaled)[0]
        result = 'Malignant' if prediction == 0 else 'Benign'
        return render_template('result.html', prediction=result)
    except ValueError:
        return render_template('result.html', prediction="Error: Please enter valid numeric values for all fields.")
    except Exception as e:
        return render_template('result.html', prediction=f"Error: An unexpected issue occurred ({str(e)}).")


if __name__ == '__main__':
    app.run(debug=True)