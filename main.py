"""
Main integration script for the AI Spam Message Classifier.
"""

import numpy as np
from load_data import load_data
from preprocess import clean_text
from features import extract_features
from model import train_model
from evaluate import evaluate_model
from predict import predict_message


def main() -> None:
    """
    Run the complete spam classification pipeline.
    """
    # Step 1: Load data
    data = load_data("../data/spam.csv")

    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Preprocess text
    data["cleaned_message"] = data["message"].apply(clean_text)

    # Step 3: Feature extraction
    features, vectorizer = extract_features(
        data["cleaned_message"].tolist()
    )

    # Step 4: Train model
    labels = np.array(data["label"])
    model, x_test, y_test = train_model(features, labels)

    # Step 5: Evaluate model
    evaluate_model(model, x_test, y_test)

    # Step 6: Predict new message
    new_message = "Congratulations! You won a free lottery prize."
    result = predict_message(model, vectorizer, new_message)

    print("\nNew Message Prediction")
    print("----------------------")
    print("Message :", new_message)
    print("Result  :", result)


if __name__ == "__main__":
    main()
