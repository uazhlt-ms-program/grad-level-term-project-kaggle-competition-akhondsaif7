"""
Train a classifier to identify reviews and generate prediction artifacts.
Author: Akhond Saif Al Masud
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def read_csv_checked(path: Path) -> pd.DataFrame:
    """Reads a CSV file and verify that required text-classification columns exist.

    Args:
        path: Path to the CSV file

    Returns:
        A pandas DataFrame containing the CSV contents

    Raises:
        ValueError: If the CSV is missing one or more required columns
    """
    df = pd.read_csv(path)
    required = {"ID", "TEXT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def select_text(df: pd.DataFrame) -> pd.Series:
    """Extracts the text column from a DataFrame for vectorization.

    Args:
        df: Input DataFrame with a TEXT column.

    Returns:
        The TEXT column as a pandas Series.
    """
    return df["TEXT"]


def build_pipeline(max_features: int = 120000, C: float = 4.0) -> Pipeline:
    """Builds the text-classification pipeline.

    The pipeline selects the text column, converts text to TF-IDF features,
    and trains a logistic regression classifier.

    Args:
        max_features: Maximum number of TF-IDF features to keep.
        C: Inverse regularization strength for logistic regression.

    Returns:
        A configured scikit-learn Pipeline.
    """
    return Pipeline(
        steps=[
            ("select_text", FunctionTransformer(select_text, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    analyzer="word",
                    max_features=max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="saga",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    """Train, tune, evaluate, and save the best review classification model."""

    parser = argparse.ArgumentParser(
        description="Train a movie/TV review classifier."
    )
    parser.add_argument("--train", default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", default="test.csv", help="Path to test.csv")
    parser.add_argument("--outdir", default="output", help="Directory for artifacts")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--n-iter", type=int, default=15, help="Random search iterations"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = read_csv_checked(Path(args.train))
    test_df = read_csv_checked(Path(args.test))

    if "LABEL" not in train_df.columns:
        raise ValueError("train.csv must contain LABEL column")

    train_df["TEXT"] = train_df["TEXT"].fillna("")
    test_df["TEXT"] = test_df["TEXT"].fillna("")

    X = train_df[["TEXT"]]
    y = train_df["LABEL"].astype(int)

    # Keep a holdout split untouched during hyperparameter tuning.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    pipeline = build_pipeline()

    param_dist = {
        "tfidf__max_features": [50000, 80000, 120000, 160000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": np.logspace(-1, 1.2, 10),
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42,
    )

    # Tune only on the training split.
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    # Evaluate once on the untouched validation split.
    valid_pred = best_pipeline.predict(X_valid)
    valid_f1 = f1_score(y_valid, valid_pred, average="macro")

    report = classification_report(y_valid, valid_pred, digits=4)
    cm = confusion_matrix(y_valid, valid_pred)

    # Retrain the selected model on all labeled data before test prediction.
    best_pipeline.fit(X, y)

    test_pred = best_pipeline.predict(test_df[["TEXT"]])

    submission = pd.DataFrame(
        {
            "ID": test_df["ID"],
            "LABEL": test_pred.astype(int),
        }
    )
    submission.to_csv(outdir / "submission.csv", index=False)

    joblib.dump(best_pipeline, outdir / "model.joblib")

    metrics = pd.DataFrame(
        {
            "metric": [
                "best_cv_f1_macro",
                "holdout_f1_macro",
            ],
            "value": [
                search.best_score_,
                valid_f1,
            ],
        }
    )
    metrics.to_csv(outdir / "metrics.csv", index=False)

    with open(outdir / "best_params.txt", "w", encoding="utf-8") as f:
        f.write(str(search.best_params_))

    with open(outdir / "report.txt", "w", encoding="utf-8") as f:
        f.write("Best parameters:\n")
        f.write(str(search.best_params_))
        f.write("\n\n")
        f.write(f"Best CV F1 macro (train only): {search.best_score_:.5f}\n")
        f.write(f"Holdout F1 macro (true generalization): {valid_f1:.5f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")

    print(f"Best CV F1 macro (train only): {search.best_score_:.5f}")
    print(f"Holdout F1 macro (true generalization): {valid_f1:.5f}")
    print(f"Wrote: {outdir / 'submission.csv'}")
    print(f"Wrote: {outdir / 'model.joblib'}")
    print(f"Wrote: {outdir / 'metrics.csv'}")
    print(f"Wrote: {outdir / 'best_params.txt'}")
    print(f"Wrote: {outdir / 'report.txt'}")


if __name__ == "__main__":
    main()
