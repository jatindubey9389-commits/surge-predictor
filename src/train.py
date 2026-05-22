"""Train XGBoost surge classifier and save to models/surge_model.pkl."""

import pathlib
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from features import build_features


RAW_PATH = pathlib.Path(__file__).parent.parent / "data" / "raw.parquet"
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "surge_model.pkl"
ZONE_FARE_MAP_PATH = MODEL_DIR / "zone_fare_map.pkl"

FEATURES = [
    "hour",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_rush_hour",
    "trip_duration_minutes",
    "pickup_zone",
    "passenger_count",
    "trip_distance",
    "zone_median_fare",
]


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data, engineer features, return X and y.

    As a side effect, writes zone_fare_map.pkl so inference can look up
    per-zone median fares without the training dataset.
    """
    print("Loading data …")
    df = pd.read_parquet(RAW_PATH)
    df = build_features(df)

    # Persist zone-level fare lookup for inference
    zone_fare_map: dict = df.groupby("PULocationID")["fare_amount"].median().to_dict()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(ZONE_FARE_MAP_PATH, "wb") as f:
        pickle.dump(zone_fare_map, f)

    available = [c for c in FEATURES if c in df.columns]
    X = df[available].copy()

    # XGBoost needs numeric; encode categorical
    for col in X.select_dtypes(["category", "object"]).columns:
        X[col] = X[col].cat.codes if hasattr(X[col], "cat") else X[col].astype("category").cat.codes

    y = df["surge_label"]
    return X, y


def train() -> None:
    X, y = load_and_prepare()

    print(f"Dataset: {X.shape[0]:,} rows | surge rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    print(f"Class balance — neg: {neg:,}  pos: {pos:,}  scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("Training XGBoost …")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
