"""Audit the supplied feature matrices and illustrate why results may be high."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from compare_models import read_pair

ROOT = Path(__file__).resolve().parent


def main():
    original_train, original_test = read_pair("original")
    engineered_train, engineered_test = read_pair("engineered")
    messages = []
    for name, tr, te in [("original", original_train, original_test),
                         ("engineered", engineered_train, engineered_test)]:
        predictors = tr.columns.drop("loan_status").tolist()
        overlap = pd.merge(tr[predictors].drop_duplicates(), te[predictors].drop_duplicates())
        messages.append(f"{name}: train={len(tr)}, test={len(te)}, exact predictor overlap={len(overlap)}")
        messages.append(f"{name}: train missing={tr.isna().sum().sum()}, test missing={te.isna().sum().sum()}")
        messages.append(f"{name}: class distribution train={tr.loan_status.value_counts().to_dict()}, "
                        f"test={te.loan_status.value_counts().to_dict()}")
    features = engineered_train.columns.drop("loan_status").tolist()
    for used in [["cibil_score"], features]:
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
        tree.fit(engineered_train[used], engineered_train.loan_status)
        pred = tree.predict(engineered_test[used])
        messages.append(f"Depth-4 tree on {used}: test accuracy={accuracy_score(engineered_test.loan_status, pred):.6f}; "
                        f"confusion [[TN,FP],[FN,TP]]={confusion_matrix(engineered_test.loan_status, pred).tolist()}")
        if len(used) > 1:
            messages.append("Tree rules (scaled terms/CIBIL):\n" + export_text(tree, feature_names=used))
    output = "\n".join(messages) + "\n"
    print(output)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_audit.txt").write_text(output)


if __name__ == "__main__":
    main()
