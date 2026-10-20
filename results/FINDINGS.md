# Results on supplied processed CSVs

Fixed split: 3,415 train, 854 test; Approved=1 and Rejected=0. The main ten models did not use sampling. Logistic Regression and RBF SVM used training-fitted scaling within each pipeline. Tree models used the supplied numerical values as-is.

| Features | Model | Accuracy | Approved precision | Approved recall | Rejected precision | Rejected recall | ROC-AUC | Confusion `[[TN,FP],[FN,TP]]` |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Original | Logistic Regression | 92.39% | 92.67% | 95.29% | 91.88% | 87.62% | 0.9745 | `[[283,40],[25,506]]` |
| Original | SVM (RBF) | 94.85% | 95.18% | 96.61% | 94.29% | 91.95% | 0.9860 | `[[297,26],[18,513]]` |
| Original | Random Forest | 98.36% | 97.96% | 99.44% | 99.05% | 96.59% | 0.9979 | `[[312,11],[3,528]]` |
| Original | AdaBoost | 97.89% | 97.59% | 99.06% | 98.41% | 95.98% | 0.9958 | `[[310,13],[5,526]]` |
| Original | XGBoost | 98.01% | 97.77% | 99.06% | 98.42% | 96.28% | 0.9985 | `[[311,12],[5,526]]` |
| Engineered | Logistic Regression | 91.57% | 91.96% | 94.73% | 90.88% | 86.38% | 0.9745 | `[[279,44],[28,503]]` |
| Engineered | SVM (RBF) | 95.08% | 94.86% | 97.36% | 95.47% | 91.33% | 0.9882 | `[[295,28],[14,517]]` |
| Engineered | Random Forest | 99.88% | 99.81% | 100% | 100% | 99.69% | 1.0000* | `[[322,1],[0,531]]` |
| Engineered | AdaBoost | 99.88% | 99.81% | 100% | 100% | 99.69% | 0.9999 | `[[322,1],[0,531]]` |
| Engineered | XGBoost | 99.88% | 100% | 99.81% | 99.69% | 100% | 1.0000* | `[[323,0],[1,530]]` |

Optional, separately trained RF + SMOTENC: 97.89% original, 100% engineered on this test set. See `model_comparison.csv` for full precision, recall, F1 and unrounded ROC-AUC.

**Audit:** No missing values and no exact feature-row overlap between supplied training and test sets. A depth-four decision tree fitted on engineered training data classified all 854 test records correctly. CIBIL alone reached 96.37% test accuracy. This points to a very simple relationship between the supplied features and labels; it does not establish how the dataset was generated or prove there is no subtler leakage. Ratios `dti` and `atr` are based on application fields; no target was passed as a predictor in these scripts. The evaluation is one fixed holdout, and comparing many models on it can make the chosen score optimistic. Real lending data and a genuinely independent test set would be needed to assess generalization.

**Deployment metrics pending raw CSV:** The four supplied files contain standardized applicant fields. `train_model.py` needs the original Kaggle CSV to train and validate an app model on rupee inputs. Its saved model's metrics will be in `deployed_model_metrics.json` after that run; the benchmark figures above are **not** automatically the saved model's result.

\* Rounded to four decimals; inspect the CSV for the precise value.
