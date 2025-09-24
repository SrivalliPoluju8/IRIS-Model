from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'knn_iris_model.pkl')
model = joblib.load(model_path)

# Serve HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON.'}), 400

        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': "Missing 'features' in request body."}), 400

        features = data['features']

        if not isinstance(features, list) or len(features) != 4:
            return jsonify({'error': 'Expected a list of 4 numerical features.'}), 400

        # Convert to NumPy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(features_array)
        predicted_class = int(prediction[0])

        # Class name (optional)
        class_map = {
            0: 'Setosa',
            1: 'Versicolor',
            2: 'Virginica'
        }

        return jsonify({
            'predicted_class': predicted_class,
            'species': class_map.get(predicted_class, 'Unknown')
        })

    except Exception as e:
        app.logger.exception("Prediction error:")
        return jsonify({'error': str(e)}), 500

# Health check (optional)
@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)







