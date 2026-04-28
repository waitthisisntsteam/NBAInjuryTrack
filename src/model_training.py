import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from config import (
    CLEANED_TRAINING_DATA_PATH,
    MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_MAX_DEPTH,
    MODEL_LEARNING_RATE,
    MODEL_N_ESTIMATORS,
    FEATURES,
    TARGET,
    CLASS_LABELS,
)


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(CLEANED_TRAINING_DATA_PATH)

    # --- 1. PREPARE THE DATA ---
    print("Preparing features and target...")
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    categorical_features = [
        "Height_Tier",
        "Weight_Tier",
        "BMI_Tier",
        "Position_Group",
    ]
    numeric_features = [col for col in FEATURES if col not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Split the data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

    # --- 2. TRAIN THE XGBOOST MODEL ---
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        eval_metric="mlogloss",
        max_depth=MODEL_MAX_DEPTH,
        learning_rate=MODEL_LEARNING_RATE,
        n_estimators=MODEL_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    print("Computing sample weights to handle class imbalance...")
    raw_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Cap the maximum weight penalty at 150x so it can surface rare classes
    sample_weights = np.clip(raw_weights, a_min=None, a_max=150.0)

    clf.fit(X_train, y_train, model__sample_weight=sample_weights)
    # --- 3. EVALUATE THE MODEL ---
    print("\nEvaluating Model Accuracy...")

    y_pred = clf.predict(X_test)

    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    label_order = sorted(CLASS_LABELS.keys())
    print(
        classification_report(
            y_test,
            y_pred,
            labels=label_order,
            target_names=[CLASS_LABELS[label] for label in label_order],
            zero_division=0,
        )
    )

    print("\n--- Probability Peek (Top 5 Highest Risk Predictions) ---")
    # Get the raw probability percentages for all classes
    probs = clf.predict_proba(X_test)

    # Find the rows where the sum of injury probabilities (Classes 1-5) is the highest
    injury_risks = np.sum(probs[:, 1:], axis=1)
    top_5_indices = np.argsort(injury_risks)[-5:][::-1]

    for idx in top_5_indices:
        print(f"\nTest Row {idx}:")
        for class_idx, class_name in CLASS_LABELS.items():
            print(f"  {class_name}: {probs[idx][class_idx]*100:.2f}%")
    # --- 4. EXPORT THE MODEL ---
    print("\nSaving the model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print("Success! Model saved as 'xgboost_injury_model.pkl'.")


if __name__ == "__main__":
    main()
