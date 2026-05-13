from fastapi import FastAPI

from pydantic import BaseModel

from predictor import predict


# create FastAPI app

app = FastAPI(

    title="Heart Disease Prediction API",

    version="1.0.0"
)


# input schema matching training features

class HeartDiseaseInput(BaseModel):

    age: int

    sex: int

    cp: int

    trestbps: float

    chol: float

    fbs: int

    restecg: int

    thalach: float

    exang: int

    oldpeak: float

    slope: int

    ca: int

    thal: int


# health check endpoint

@app.get("/health")

def health_check():

    return {

        "status": "ok"
    }


# heart disease prediction endpoint

@app.post("/predict-heart-disease")

def predict_heart_disease(
    input_data: HeartDiseaseInput
):

    result = predict(
        input_data=input_data.dict()
    )

    return {

        "prediction": result["prediction"],

        "probability": result["probability"],

        "diagnosis": (

            "Heart Disease Detected"

            if result["prediction"] == 1

            else "No Heart Disease Detected"
        )
    }