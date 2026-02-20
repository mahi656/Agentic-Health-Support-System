import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


def train_models(file):
    df = pd.read_csv(file)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    tree_pipeline = Pipeline([
        ("model", DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            random_state=42
        ))
    ])

    logistic_pipeline.fit(X_train, y_train)
    tree_pipeline.fit(X_train, y_train)

    logistic_metrics = evaluate_model(logistic_pipeline, X_test, y_test)
    tree_metrics = evaluate_model(tree_pipeline, X_test, y_test)

    logistic_cv_accuracy = cross_val_score(
        logistic_pipeline, X, y, cv=5, scoring="accuracy"
    ).mean()

    tree_cv_accuracy = cross_val_score(
        tree_pipeline, X, y, cv=5, scoring="accuracy"
    ).mean()

    if logistic_metrics["f1_score"] > tree_metrics["f1_score"]:
        best_model = logistic_pipeline
        best_name = "Logistic Regression"
    else:
        best_model = tree_pipeline
        best_name = "Decision Tree"

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    return {
        "logistic_metrics": logistic_metrics,
        "tree_metrics": tree_metrics,
        "logistic_cv_accuracy": logistic_cv_accuracy,
        "tree_cv_accuracy": tree_cv_accuracy,
        "best_model": best_name
    }