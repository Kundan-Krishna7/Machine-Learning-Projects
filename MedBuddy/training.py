import os
import logging

import pandas as pd

from pathlib import Path

from dotenv import load_dotenv

from joblib import dump

from sklearn.model_selection import GroupShuffleSplit

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
)


def train_model():

    try:

        # load env variables

        load_dotenv()

        # project paths

        PROJECT_ROOT = Path.cwd()

        DATASET_PATH = PROJECT_ROOT / "heart.csv"

        MODEL_PATH = PROJECT_ROOT / "heart_model.joblib"

        LOG_PATH = PROJECT_ROOT / "training.log"

        # logging configuration

        logging.basicConfig(
            filename=LOG_PATH,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        print("Loading dataset...")

        # load dataset

        df = pd.read_csv(DATASET_PATH)

        print("Dataset loaded successfully ✅")

        # features and target

        X = df.drop("target", axis=1)

        y = df["target"]

        # create groups

        groups = pd.util.hash_pandas_object(
            X,
            index=False
        )

        print("Groups created successfully ✅")

        # group split

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42
        )

        train_idx, test_idx = next(
            gss.split(X, y, groups=groups)
        )

        # train test data

        X_train, X_test = (
            X.iloc[train_idx],
            X.iloc[test_idx]
        )

        y_train, y_test = (
            y.iloc[train_idx],
            y.iloc[test_idx]
        )

        print("Train-Test Split Completed ✅")

        print("Train Shape:", X_train.shape)

        print("Test Shape:", X_test.shape)

        # pipeline

        pipeline = Pipeline(
            steps=[

                ("scaler", StandardScaler()),

                ("model", RandomForestClassifier(
                    random_state=42,
                    n_estimators=500,
                    max_depth=6,
                    n_jobs=-1
                ))
            ]
        )

        print("Training model...")

        # train model

        pipeline.fit(X_train, y_train)

        print("Model training completed ✅")

        # predictions

        y_pred = pipeline.predict(X_test)

        # metrics

        acc = accuracy_score(y_test, y_pred)

        rec = recall_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        print(f"\nAccuracy Score : {acc:.2%}")

        print(f"Recall Score   : {rec:.2%}")

        print(f"F1 Score       : {f1:.2%}")

        print("\nClassification Report\n")

        print(classification_report(y_test, y_pred))

        # save model

        dump(pipeline, MODEL_PATH)

        print("\nModel saved successfully ✅")

        logging.info(
            "Training completed successfully"
        )

    except Exception as e:

        print(f"Training failed: {e}")

        logging.exception(
            f"Training Script Failed: {e}"
        )

        raise


if __name__ == "__main__":

    train_model()