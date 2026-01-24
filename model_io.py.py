"""
Model input/output module for saving and loading trained models.
"""

import pickle
from typing import Any


def save_model(model: Any, vectorizer: Any, model_path: str) -> None:
    """
    Save trained model and vectorizer to disk.

    Args:
        model (Any): Trained ML model
        vectorizer (Any): Fitted text vectorizer
        model_path (str): File path to save model
    """
    with open(model_path, "wb") as file:
        pickle.dump(
            {"model": model, "vectorizer": vectorizer},
            file
        )

    print(f"Model saved to {model_path}")


def load_model(model_path: str) -> Any:
    """
    Load trained model and vectorizer from disk.

    Args:
        model_path (str): File path to load model

    Returns:
        Any: Dictionary containing model and vectorizer
    """
    with open(model_path, "rb") as file:
        data = pickle.load(file)

    print(f"Model loaded from {model_path}")
    return data


def main() -> None:
    """
    Test model saving and loading with dummy objects.
    """

    class DummyModel:
        """Simple dummy model."""
        pass

    class DummyVectorizer:
        """Simple dummy vectorizer."""
        pass

    dummy_model = DummyModel()
    dummy_vectorizer = DummyVectorizer()

    save_model(dummy_model, dummy_vectorizer, "dummy_model.pkl")
    loaded = load_model("dummy_model.pkl")

    print("Loaded keys:", loaded.keys())


if __name__ == "__main__":
    main()
