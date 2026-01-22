"""
Prediction module for spam message classification.
"""

from typing import Any


def predict_message(model: Any, vectorizer: Any, message: str) -> str:
    """
    Predict whether a message is spam or not spam.

    Args:
        model (Any): Trained ML model
        vectorizer (Any): Fitted text vectorizer
        message (str): Input text message

    Returns:
        str: Prediction result ("SPAM" or "NOT SPAM")
    """
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        return "SPAM"

    return "NOT SPAM"


def main() -> None:
    """
    Test prediction function with dummy objects.
    """

    class DummyVectorizer:
        """Simple dummy vectorizer."""

        def transform(self, texts: Any) -> Any:
            return texts

    class DummyModel:
        """Simple dummy model."""

        def predict(self, _: Any) -> list:
            return [1]

    dummy_vectorizer = DummyVectorizer()
    dummy_model = DummyModel()

    test_message = "Win a free prize now!"
    result = predict_message(dummy_model, dummy_vectorizer, test_message)

    print("Message :", test_message)
    print("Prediction:", result)


if __name__ == "__main__":
    main()
