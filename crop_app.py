from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)
app.secret_key = 'crop_recommendation_secret_key'

# Load and prepare the data
def prepare_model():
    try:
        # Try to load existing model
        if os.path.exists('crop_model.pkl'):
            return joblib.load('crop_model.pkl')
        
        # If model doesn't exist, train a new one
        df = pd.read_csv('Crop_recommendation.csv')
        
        # Prepare features and target
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, 'crop_model.pkl')
        
        return model
    except Exception as e:
        print(f"Error preparing model: {e}")
        return None

# Load the model
model = prepare_model()

# Get unique crops for reference
try:
    df = pd.read_csv('Crop_recommendation.csv')
    CROPS = sorted(df['label'].unique())
except:
    CROPS = []

@app.route('/')
def home():
    return render_template('crop_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        n = float(request.form['nitrogen'])
        p = float(request.form['phosphorus'])
        k = float(request.form['potassium'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Make prediction
        features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        prediction = model.predict(features)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(features)[0]
        # Get top 3 recommendations with probabilities
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        recommendations = [
            (model.classes_[idx], round(probabilities[idx] * 100, 2))
            for idx in top_3_idx
        ]
        
        return render_template(
            'crop_result.html',
            prediction=prediction,
            recommendations=recommendations,
            input_values={
                'Nitrogen': n,
                'Phosphorus': p,
                'Potassium': k,
                'Temperature': temp,
                'Humidity': humidity,
                'pH': ph,
                'Rainfall': rainfall
            }
        )
        
    except Exception as e:
        flash(f'Error making prediction: {str(e)}')
        return render_template('crop_index.html')

@app.route('/about')
def about():
    return render_template('crop_about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000) 