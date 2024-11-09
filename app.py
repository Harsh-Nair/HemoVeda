from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sys
from sklearn.ensemble import RandomForestClassifier
import joblib

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)


palm_model = load_model('/path/to/your/saved/trained/model-or-weights')
conjunctiva_model = load_model('/path/to/your/saved/trained/model-or-weights')


try:
    blood_report_model = joblib.load('/path/to/your/saved/trained/model-or-weights')
except Exception as e:
    print(f"Error loading the model: {e}")
    blood_report_model = None


UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict_palm', methods=['POST'])
def predict_palm():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = palm_model.predict(image)
        confidence = np.max(predictions)
        result = "Anemic" if confidence <= 0.5 else "Healthy"

        os.remove(filepath)

        return jsonify({
            'prediction': result,
            'confidence_percentage': f"{confidence * 100:.2f}%"
        }), 200, {'Content-Type': 'application/json; charset=utf-8'}

    return jsonify({'error': 'Invalid file type. Please upload a valid image file.'}), 400

    

@app.route('/predict_conjunctiva', methods=['POST'])
def predict_conjunctiva():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = conjunctiva_model.predict(image)
        confidence = np.max(predictions)
        result = "Anemic" if confidence <= 0.5 else "Healthy"

        os.remove(filepath)

        return jsonify({
            'prediction': result,
            'confidence_percentage': f"{confidence * 100:.2f}%"
        }), 200, {'Content-Type': 'application/json; charset=utf-8'}

    return jsonify({'error': 'Invalid file type. Please upload a valid image file.'}), 400


   

@app.route('/predict_blood_report', methods=['POST'])
def predict_blood_report():
    if blood_report_model is None:
        return jsonify({'error': 'Blood report model is not available'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Extract features from the input data
        features = np.array([
            float(data['gender']),
            float(data['haemoglobin']),
            float(data['mch']),
            float(data['mchc']),
            float(data['mcv'])
        ]).reshape(1, -1)

        # Make prediction using the blood report model
        prediction = blood_report_model.predict(features)
        prediction_proba = blood_report_model.predict_proba(features)
        
        confidence = np.max(prediction_proba) * 100
        result = "Anemic" if prediction[0] == 1 else "Healthy"

        return jsonify({
            'prediction': result,
            'confidence_percentage': f"{confidence:.2f}%"
        }), 200, {'Content-Type': 'application/json; charset=utf-8'}

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)