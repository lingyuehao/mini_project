from __future__ import annotations

import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score

__all__ = ["timed_execution", "check_missing", "evaluate_model"]


def timed_execution(func, *args, **kwargs):
    """
    Run a function and print elapsed time in seconds.
    Returns the function result unchanged.
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} completed in {end - start:.6f} seconds")
    return result


def check_missing(df):
    """
    Print per-column and total missing counts for pandas or polars DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        print("\n--- Missing Values per Column ---")
        print(df.isnull().sum())
        print("\n--- Total Missing Values ---")
        print(df.isnull().sum().sum())
    elif isinstance(df, pl.DataFrame):
        print("\n--- Missing Values per Column ---")
        print(df.null_count())
        print("\n--- Total Missing Values ---")
        print(df.null_count().to_numpy().sum())
    else:
        raise TypeError("df must be a pandas or polars DataFrame")


def evaluate_model(model, X_test, y_test, features):
    """
    Predict, print RMSE/R², and show feature importances (sklearn tree models).
    Returns (rmse, r2, importance_df) for convenience.
    """
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print("\n--- Model Evaluation ---")
    print("RMSE:", round(rmse, 3))
    print("R²:", round(r2, 3))

    if hasattr(model, "feature_importances_"):
        importance = (
            pd.DataFrame(
                {"Feature": list(features), "Importance": model.feature_importances_}
            )
            .sort_values(by="Importance", ascending=False)
            .reset_index(drop=True)
        )
        print("\nFeature Importances:")
        print(importance)
    else:
        importance = pd.DataFrame(columns=["Feature", "Importance"])

    return rmse, r2, importance
