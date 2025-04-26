from flask import Flask, render_template, request, flash, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'crop_recommendation_secret_key'

# Global variables for model and crops
model = None
CROPS = []

def prepare_model():
    """Prepare and load the crop recommendation model"""
    global model, CROPS
    try:
        model_path = Path('crop_model.pkl')
        
        # Try to load existing model
        if model_path.exists():
            model = joblib.load(model_path)
            return model
        
        # If model doesn't exist, train a new one
        df = pd.read_csv('Crop_recommendation.csv')
        
        # Prepare features and target
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model with improved parameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, model_path)
        
        # Update CROPS list
        CROPS = sorted(df['label'].unique())
        
        return model
    except Exception as e:
        print(f"Error preparing model: {e}")
        return None

# Initialize model and crops
model = prepare_model()

@app.route('/')
def home():
    return render_template('crop_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form with input validation
        input_data = {
            'nitrogen': float(request.form['nitrogen']),
            'phosphorus': float(request.form['phosphorus']),
            'potassium': float(request.form['potassium']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        # Validate input ranges
        if not all(0 <= input_data[param] <= 1000 for param in ['nitrogen', 'phosphorus', 'potassium']):
            raise ValueError("NPK values should be between 0 and 1000")
        if not 0 <= input_data['ph'] <= 14:
            raise ValueError("pH value should be between 0 and 14")
        
        # Make prediction
        features = np.array([[input_data[param] for param in input_data]])
        prediction = model.predict(features)[0]
        
        # Get probability scores
        probabilities = model.predict_proba(features)[0]
        # Get top 3 recommendations with probabilities
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        recommendations = [
            {
                'crop': model.classes_[idx],
                'probability': round(probabilities[idx] * 100, 2),
                'confidence': 'High' if probabilities[idx] > 0.5 else 'Medium' if probabilities[idx] > 0.3 else 'Low'
            }
            for idx in top_3_idx
        ]
        
        return render_template(
            'crop_result.html',
            prediction=prediction,
            recommendations=recommendations,
            input_values=input_data
        )
        
    except ValueError as ve:
        flash(f'Invalid input: {str(ve)}', 'error')
        return render_template('crop_index.html')
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'error')
        return render_template('crop_index.html')

@app.route('/about')
def about():
    return render_template('crop_about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Make prediction using the same logic as the web interface
        features = np.array([[
            float(data.get('nitrogen', 0)),
            float(data.get('phosphorus', 0)),
            float(data.get('potassium', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            float(data.get('ph', 0)),
            float(data.get('rainfall', 0))
        ]])
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': prediction,
            'probabilities': {
                model.classes_[i]: round(prob * 100, 2)
                for i, prob in enumerate(probabilities)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)