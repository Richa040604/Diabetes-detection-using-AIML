import numpy as np
import pickle

# Load the trained model and the scaler
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to take input data and test the model
def test_model():
    print("Enter the following values for diabetes prediction:")
    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))

    # Create an array from the input data
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    input_data = input_data.reshape(1, -1)

    # Scale the input data using the pre-fitted scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data_scaled)

    # Print the prediction result
    if prediction[0] == 1:
        print("The model predicts that the person is likely to have diabetes.")
    else:
        print("The model predicts that the person is unlikely to have diabetes.")

# Test the model with user input
test_model()
