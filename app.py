import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained heart disease prediction model
heart_model = joblib.load("heart_disease_model.joblib")

# Create FastAPI instance
app = FastAPI()

# Define the expected input schema using Pydantic BaseModel
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Home route with a welcome message
@app.get('/')
async def welcome():
    return {'message': 'Welcome to the Heart Disease Prediction API!'}

# Prediction route
@app.post('/predict')
async def predict_heart_disease(input_data: HeartDiseaseInput):
    # Convert input data to dictionary format
    input_dict = input_data.dict()

    # Prepare input data for the model
    feature_values = [
        input_dict['age'], input_dict['sex'], input_dict['cp'], input_dict['trestbps'], 
        input_dict['chol'], input_dict['fbs'], input_dict['restecg'], input_dict['thalach'], 
        input_dict['exang'], input_dict['oldpeak'], input_dict['slope'], input_dict['ca'], 
        input_dict['thal']
    ]

    # Convert the features to a numpy array and reshape it for prediction
    features_array = np.array(feature_values).reshape(1, -1)

    # Perform prediction using the preloaded model
    prediction = heart_model.predict(features_array)

    # Map the prediction result to a human-readable message
    diagnosis = 'The person has Heart Disease' if prediction[0] == 1 else 'The person does not have Heart Disease'

    # Return the prediction result
    return {'prediction': diagnosis}

# Start the FastAPI server
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
