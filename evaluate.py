"""
Model evaluation module for spam message classification.
"""

from typing import Any
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import numpy as np


def evaluate_model(
    model: Any,
    x_test: Any,
    y_test: np.ndarray
) -> None:
    """
    Evaluate the trained model using standard metrics.

    Args:
        model (Any): Trained ML model
        x_test (Any): Test feature matrix
        y_test (np.ndarray): True labels
    """
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("Model Evaluation Results")
    print("------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", matrix)


def main() -> None:
    """
    Test evaluation function with dummy data.
    """
    # Dummy example
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])

    class DummyModel:
        """Simple dummy model for testing."""

        def predict(self, _: Any) -> np.ndarray:
            return y_pred

    dummy_model = DummyModel()

    evaluate_model(dummy_model, None, y_true)


if __name__ == "__main__":
    main()
