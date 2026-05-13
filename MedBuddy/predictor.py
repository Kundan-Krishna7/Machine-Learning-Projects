import os
import logging

from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

from joblib import load


# load .env content to env vars

load_dotenv()


# project paths

PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "heart_model.joblib"

LOG_PATH = PROJECT_ROOT / "prediction.log"


# create log directory if needed

LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# logging configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)


# load trained model only once

model = load(MODEL_PATH)

logging.info("Model loaded successfully ")


# prediction function

def predict(input_data: dict):

    try:

        # convert dictionary to dataframe

        df = pd.DataFrame([input_data])

        # prediction

        prediction = int(model.predict(df)[0])

        # probability

        probability = float(
            model.predict_proba(df)[0][1]
        )

        logging.info(
            f"Model provided prediction: {prediction}, "
            f"probability: {probability}"
        )

        # return result

        return {

            "prediction": prediction,

            "probability": probability
        }

    except Exception as e:

        logging.exception(
            f"Prediction failed: {e}"
        )

        return {

            "error": str(e)
        }


# example usage

#