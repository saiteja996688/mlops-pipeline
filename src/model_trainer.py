"""Model training pipeline for MLOps workflows."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


class ModelTrainer:
    """Handles model training, evaluation, and persistence."""

    def __init__(self, model_dir="models", scaler_dir="scalers"):
        self.model_dir = model_dir
        self.scaler_dir = scaler_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(scaler_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None

    def preprocess(self, df, target_col):
        """
        Split features and target, then scale features.

        Args:
            df: pandas DataFrame with features and target
            target_col: Name of the target column

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X, y, model_type="random_forest"):
        """
        Train a classification model.

        Args:
            X: Feature matrix (numpy array)
            y: Target labels
            model_type: Either "logistic" or "random_forest"
        """
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.model.fit(X, y)
        return self.model

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            dict: Accuracy and classification report
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {"accuracy": accuracy, "report": report}

    def save_model(self, model_name="model_v1"):
        """Save trained model and scaler to disk."""
        joblib.dump(self.model, os.path.join(self.model_dir, f"{model_name}.joblib"))
        joblib.dump(self.scaler, os.path.join(self.scaler_dir, "scaler.joblib"))
        print(f"Model saved to {self.model_dir}/{model_name}.joblib")
        print(f"Scaler saved to {self.scaler_dir}/scaler.joblib")

    def load_model(self, model_name="model_v1"):
        """Load a trained model from disk."""
        self.model = joblib.load(os.path.join(self.model_dir, f"{model_name}.joblib"))
        self.scaler = joblib.load(os.path.join(self.scaler_dir, "scaler.joblib"))
        return self.model
