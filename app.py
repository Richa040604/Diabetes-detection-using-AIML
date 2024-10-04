# app.py
from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'], 
                           data['SkinThickness'], data['Insulin'], data['BMI'], 
                           data['DiabetesPedigreeFunction'], data['Age']])
    input_data = input_data.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    return jsonify({'diabetes_prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
