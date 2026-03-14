# task1_linear_svm_churn.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

RANDOM_STATE = 42


# -------------------------
# Hardcoded values
# -------------------------

DATA_PATH = "Telco-Customer-Churn.csv"

TARGET = "Churn"

TEST_SIZE = 0.2

C_VALUES = [0.1, 1, 10]


# -------------------------
# Preprocess
# -------------------------

def load_and_preprocess():

    df = pd.read_csv(DATA_PATH)

    customer_ids = df["customerID"]

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Encode target
    y = df[TARGET].map({
        "Yes": 1,
        "No": 0
    })

    X = df.drop(
        columns=[TARGET, "customerID"]
    )

    # One hot encoding
    X = pd.get_dummies(
        X,
        drop_first=True
    )

    # ✅ Fix remaining NaN if any
    X = X.fillna(0)

    return X, y, customer_ids


# -------------------------
# Evaluate
# -------------------------

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    scores = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, scores)

    return cm, acc, prec, rec, f1, roc, y_pred, scores


# -------------------------
# Main
# -------------------------

def main():

    X, y, customer_ids = load_and_preprocess()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    for C_value in C_VALUES:

        print("\n====================")
        print("C =", C_value)

        model = SVC(
            kernel="linear",
            C=C_value,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)

        cm, acc, prec, rec, f1, roc, y_pred, scores = evaluate_model(
            model,
            X_test,
            y_test
        )

        print("Confusion Matrix:\n", cm)
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1:", f1)
        print("ROC-AUC:", roc)

        n_sv = model.n_support_.sum()

        print("Support Vectors:", n_sv)

        results.append({
            "C": C_value,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "support_vectors": n_sv
        })

        # Save predictions
        pred_df = pd.DataFrame({
            "CustomerID": id_test,
            "Actual": y_test,
            "Predicted": y_pred,
            "Score": scores
        })

        pred_df.to_csv(
            "test_predictions.csv",
            index=False
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "svm_linear_results.csv",
        index=False
    )

    print("\nSummary Table:")
    print(results_df)


if __name__ == "__main__":
    main()