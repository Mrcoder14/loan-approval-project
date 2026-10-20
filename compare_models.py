"""Reproduce held-out benchmarks from the supplied processed CSV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
TARGET = "loan_status"
BINARY = ["education", "self_employed"]


def read_pair(kind):
    train = pd.read_csv(DATA / f"train_{kind}.csv")
    test = pd.read_csv(DATA / f"test_{kind}.csv")
    if list(train.columns) != list(test.columns) or TARGET not in train:
        raise ValueError(f"{kind}: column mismatch or missing target")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(f"{kind}: missing values; check source files")
    if set(train[TARGET].unique()) != {0, 1} or set(test[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected Approved=1 and Rejected=0")
    return train, test


def models(columns):
    numeric = [c for c in columns if c not in BINARY]
    pre = ColumnTransformer([("numeric", StandardScaler(), numeric),
                             ("binary", "passthrough", BINARY)])
    return {
        "Logistic Regression": make_pipeline(pre, LogisticRegression(max_iter=3000, random_state=42)),
        "SVM (RBF)": make_pipeline(pre, SVC(kernel="rbf", probability=True, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, n_jobs=2,
                                 eval_metric="logloss", tree_method="hist"),
    }


def score(model, x, y):
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, list(model.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "approved_precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "approved_recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "approved_f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "rejected_precision": precision_score(y, pred, pos_label=0, zero_division=0),
        "rejected_recall": recall_score(y, pred, pos_label=0, zero_division=0),
        "rejected_f1": f1_score(y, pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "tn_rejected_correct": int(tn), "fp_rejected_as_approved": int(fp),
        "fn_approved_as_rejected": int(fn), "tp_approved_correct": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smote", action="store_true", help="Also compare RF with train-only SMOTE")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    original = read_pair("original")
    engineered = read_pair("engineered")
    for a, b in zip(original, engineered):
        if len(a) != len(b) or not a[TARGET].equals(b[TARGET]):
            raise ValueError("Original and engineered rows are not aligned")
        if not np.allclose(a.drop(columns=TARGET), b[a.drop(columns=TARGET).columns]):
            raise ValueError("Original predictors changed between feature sets")

    rows = []
    for kind, (train, test) in [("original", original), ("engineered", engineered)]:
        features = [c for c in train if c != TARGET]
        x_train, y_train = train[features], train[TARGET]
        x_test, y_test = test[features], test[TARGET]
        for name, model in models(features).items():
            model.fit(x_train, y_train)
            rows.append({"features": kind, "model": name,
                         "train_accuracy": accuracy_score(y_train, model.predict(x_train)),
                         **score(model, x_test, y_test)})
            print(f"{kind:10s} {name:20s} test accuracy {rows[-1]['accuracy']:.4f} "
                  f"rejected recall {rows[-1]['rejected_recall']:.4f}")

    if args.smote:
        from imblearn.over_sampling import SMOTENC
        from imblearn.pipeline import make_pipeline as imbalanced_pipeline
        # Binary indicators require a categorical-aware sampler.
        for kind, (train, test) in [("original", original), ("engineered", engineered)]:
            features = [c for c in train if c != TARGET]
            sampler = SMOTENC(categorical_features=[features.index(c) for c in BINARY], random_state=42)
            model = imbalanced_pipeline(sampler, RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1))
            model.fit(train[features], train[TARGET])
            rows.append({"features": kind, "model": "Random Forest + SMOTENC",
                         "train_accuracy": accuracy_score(train[TARGET], model.predict(train[features])),
                         **score(model, test[features], test[TARGET])})

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "model_comparison.csv", index=False, float_format="%.8f")
    (OUT / "model_comparison.json").write_text(json.dumps(rows, indent=2))
    print("\nConfusion matrix cells: [[TN, FP], [FN, TP]]; positive class is Approved (1).")
    print(result.to_string(index=False, float_format=lambda n: f"{n:.4f}"))


if __name__ == "__main__":
    main()
