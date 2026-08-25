"""
House Price Prediction - Main Entry Point
Author: Vinnu (@vinnueditx)
"""

import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default file paths
DATA_PATH = os.path.join("data", "house_data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> Pipeline:
    """Loads dataset, trains preprocessing + Random Forest pipeline, and saves artifact."""
    if not os.path.exists(data_path):
        print(f"[-] Error: Dataset not found at '{data_path}'. Please check the path.")
        sys.exit(1)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"[+] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df):,} rows and {len(df.columns)} columns.")

    if "price" not in df.columns:
        print("[-] Error: 'price' column not found in dataset.")
        sys.exit(1)

    # Clean target & features
    df = df.dropna(subset=["price"])
    X = df.drop(columns=["price"])
    y = df["price"]

    # Detect feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[+] Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"[+] Categorical features ({len(categorical_features)}): {categorical_features}")

    # Build pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("\n[+] Training Random Forest Regressor...")
    full_pipeline.fit(X_train, y_train)
    print("[+] Training complete.")

    # Evaluate
    predictions = full_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 36)
    print("        MODEL EVALUATION           ")
    print("=" * 36)
    print(f"Mean Absolute Error : ₹{mae:,.2f}")
    print(f"Root Mean Sq Error  : ₹{rmse:,.2f}")
    print(f"R² Score            : {r2:.4f}")
    print("=" * 36)

    # Save artifact
    with open(model_path, "wb") as f:
        pickle.dump(full_pipeline, f)

    print(f"\n[✓] Pipeline successfully saved to: {model_path}")
    return full_pipeline


def predict(input_dict: dict, model_path: str = MODEL_PATH) -> float:
    """Loads serialized pipeline and performs inference on input dictionary."""
    if not os.path.exists(model_path):
        print(f"[-] Error: Model file '{model_path}' not found. Run `--train` first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    df_input = pd.DataFrame([input_dict])
    prediction = pipeline.predict(df_input)[0]
    return float(prediction)


def main():
    parser = argparse.ArgumentParser(
        description="House Price Prediction ML Pipeline - by @vinnueditx"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model on dataset and export pipeline artifact",
    )
    parser.add_argument(
        "--predict-sample",
        action="store_true",
        help="Run inference on a built-in sample house",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help="Path to CSV dataset (default: data/house_data.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to save/load model pickle (default: models/house_price_model.pkl)",
    )

    args = parser.parse_args()

    if args.train:
        train(data_path=args.data, model_path=args.model)
    elif args.predict_sample:
        # Update keys to match the column names in your house_data.csv
        sample_house = {
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft_living": 1850,
            "neighborhood": "Downtown",
            "year_built": 2015,
        }
        print(f"[+] Sample input: {sample_house}")
        price = predict(sample_house, model_path=args.model)
        print(f"\n[✓] Estimated House Price: ₹{price:,.2f}")
    else:
        # Default behavior when run with no arguments
        print("[*] No flags provided. Training model by default...\n")
        train(data_path=args.data, model_path=args.model)


if __name__ == "__main__":
    main()
