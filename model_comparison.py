"""
Model comparison for spam message classification.
Compares Naive Bayes and Logistic Regression.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from load_data import load_data
from preprocess import clean_text
from features import extract_features
from config import DATA_PATH


def compare_models() -> None:
    """
    Train and compare two ML models using accuracy.
    """
    data = load_data(DATA_PATH)

    if data is None:
        print("Failed to load dataset.")
        return

    # Preprocess text
    data["cleaned_message"] = data["message"].apply(clean_text)

    # Feature extraction
    features, _ = extract_features(
        data["cleaned_message"].tolist()
    )
    labels = np.array(data["label"])

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)
    nb_predictions = nb_model.predict(x_test)
    nb_accuracy = accuracy_score(y_test, nb_predictions)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(x_train, y_train)
    lr_predictions = lr_model.predict(x_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    print("Model Comparison Results")
    print("------------------------")
    print(f"Naive Bayes Accuracy      : {nb_accuracy:.4f}")
    print(f"Logistic Regression Acc.  : {lr_accuracy:.4f}")


if __name__ == "__main__":
    compare_models()
