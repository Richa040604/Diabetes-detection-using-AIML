import requests

# Define the endpoint and the data to send
url = 'http://127.0.0.1:5000/predict'
data = {
    "Pregnancies": 3,
    "Glucose": 150,
    "BloodPressure": 70,
    "SkinThickness": 30,
    "Insulin": 120,
    "BMI": 33.5,
    "DiabetesPedigreeFunction": 0.4,
    "Age": 50
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response (diabetes prediction)
print(response.json())
