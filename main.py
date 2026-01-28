"""
Main integration script for the AI Spam Message Classifier.
"""

import logging
import numpy as np

from load_data import load_data
from preprocess import clean_text
from features import extract_features
from model import train_model
from evaluate import evaluate_model
from predict import predict_message
from model_io import save_model, load_model
from config import DATA_PATH, MODEL_PATH, LOG_LEVEL


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run the complete spam classification pipeline.
    """
    logger.info("Starting spam classification pipeline")

    # Step 1: Load data
    data = load_data(DATA_PATH)

    if data is None:
        logger.error("Failed to load data. Exiting.")
        return

    # Step 2: Preprocess text
    logger.info("Preprocessing text data")
    data["cleaned_message"] = data["message"].apply(clean_text)

    # Step 3: Feature extraction
    logger.info("Extracting text features")
    features, vectorizer = extract_features(
        data["cleaned_message"].tolist()
    )

    # Step 4: Train model
    logger.info("Training Naive Bayes model")
    labels = np.array(data["label"])
    model, x_test, y_test = train_model(features, labels)

    # Step 5: Evaluate model
    logger.info("Evaluating model performance")
    evaluate_model(model, x_test, y_test)

    # Step 6: Save model
    logger.info("Saving trained model")
    save_model(model, vectorizer, MODEL_PATH)

    # Step 7: Load model
    logger.info("Loading trained model")
    loaded = load_model(MODEL_PATH)
    loaded_model = loaded["model"]
    loaded_vectorizer = loaded["vectorizer"]

    # Step 8: Predict new message
    new_message = "Congratulations! You won a free lottery prize."
    logger.info("Making prediction for new message")
    result = predict_message(
        loaded_model,
        loaded_vectorizer,
        new_message
    )

    logger.info("New Message: %s", new_message)
    logger.info("Prediction: %s", result)
    logger.info("Spam classification pipeline completed successfully")


if __name__ == "__main__":
    main()
