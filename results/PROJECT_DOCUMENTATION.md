# Loan Approval Prediction — Project Documentation

## 1. Project overview and problem statement

**Goal:** Build a reproducible classifier that predicts the historical `Approved` or `Rejected` label from an applicant's financial and demographic fields, compare several algorithms fairly, and demonstrate predictions in a Streamlit interface.

A manual review of many applications can be slow and inconsistent. The project explores whether patterns in the supplied dataset can support a quick, explainable demonstration of classification. **This is an academic label-prediction project:** the dataset label is not evidence of repayment, and this model is not validated for real lending decisions.

The main questions were: Which model works best on a common holdout? Do engineered ratios help? Are near-perfect results a coding mistake, data leakage, or a property of this particular dataset? Can a teammate reproduce training and inspect individual predictions?

## 2. Approach and why each step was taken

1. **Inspect the data and labels.** We checked column names, missing data, class balance, and the train/test partitions before fitting models. This prevents silent comparisons on incompatible files. The supplied 4,269 records comprise 3,415 train and 854 test rows; train labels are 2,125 Approved / 1,290 Rejected, and test labels are 531 Approved / 323 Rejected.
2. **Preserve one fixed holdout across models.** Each algorithm is evaluated on the same 854 supplied test rows so their metrics are comparable. All fitting, including scaling and optional resampling, happens on training data only.
3. **Compare simple and flexible algorithms.** Logistic Regression establishes a linear baseline; an RBF SVM captures nonlinear boundaries; Random Forest, AdaBoost, and XGBoost test tree-based relationships. We compare each on the original predictor set and the expanded engineered set, yielding ten primary experiments.
4. **Audit unexpectedly high accuracy.** We checked missing values and exact duplicate feature rows across train/test, compared a CIBIL-only tree to a shallow tree on all engineered features, and checked that the derived fields do not use the label. This rules out a few obvious errors but cannot prove generalization or rule out all leakage in the source data.
5. **Train the app model from unscaled raw data.** The four supplied processed CSVs contain standardized financial columns, which cannot be directly entered as rupees in a form. The raw CSV allows a model to accept human-readable values. `train_model.py` matches raw rows back to the supplied fixed split and verifies the correspondence before fitting.
6. **Expose and record predictions.** `app.py` takes manual inputs, shows model output and derived values, runs seven illustrative cases from a button, and stores full inputs and outputs in a local SQLite history for review.

## 3. Data sources and modifications

| File | Role | Important detail |
|---|---|---|
| `data/loan_approval_dataset.csv` | Original raw source, 4,269 rows | Text categories and financial values in original units; `loan_id` identifies records, `loan_status` is the target. |
| `data/train_original.csv`, `data/test_original.csv` | Supplied fixed split with 11 predictors | “Original” describes the feature set; several numeric values have **already been standardized**. These are not rupee inputs. |
| `data/train_engineered.csv`, `data/test_engineered.csv` | Same aligned split, with 14 predictors | Add `total_asset_value`, `dti`, and `atr` to the original 11 columns. |

The classifier maps `Approved → 1` and `Rejected → 0`. Education maps `Graduate → 1`, `Not Graduate → 0`; self-employment maps `Yes → 1`, `No → 0`. `loan_id` and `loan_status` are excluded from input features. `features.py` is the shared definition of transformations for raw-data training and app inference, avoiding differences between the two paths.

In the supplied processed files, selected numeric original columns were standardized using training-set statistics. For a value \(x\), training mean \(\mu_{train}\), and training standard deviation \(\sigma_{train}\):

\[
z = \frac{x - \mu_{train}}{\sigma_{train}}.
\]

The scaler must learn these statistics from training rows; fitting on test rows would leak evaluation information. `train_model.py` independently fits a training-only scaler to matched raw rows and checks that its outputs agree with the supplied standardized columns. The **deployed** Random Forest instead uses raw numeric values and does not apply a scaler.

### Engineered mathematical features

Given residential, commercial, luxury, and bank assets \(R,C,L,B\), annual income \(I\), and requested loan \(Q\):

\[
\text{total_asset_value}=R+C+L+B,\qquad
\text{dti}=\frac{Q}{I},\qquad
\text{atr}=\frac{R+C+L+B}{Q}.
\]

- **Total assets** aggregates the four asset columns so the model can inspect overall asset size.
- **`dti`** measures requested loan divided by annual income. The project retains this existing column name for compatibility, but it is **not conventional debt-to-income**, which requires actual recurring debt payments. It is better interpreted as a *loan-to-income ratio*.
- **`atr`** measures listed assets relative to requested loan. It helps expose a ratio relationship that may otherwise require several tree splits.

`features.py` rejects incomplete records and nonpositive income or loan amount, avoiding undefined division. These features are deterministic functions of applicant inputs; they do not read `loan_status`. The already processed engineered CSVs are supplied artifacts; our scripts validate their alignment and reproduce these feature definitions when training the app model. We cannot infer every historical preprocessing choice from the CSVs alone.

## 4. Model selection and mathematical ideas

| Model | Why include it | Main idea |
|---|---|---|
| Logistic Regression | Interpretable linear baseline | \(P(y=1\mid x)=1/(1+e^{-(w^Tx+b)})\). |
| SVM with RBF kernel | Flexible nonlinear baseline | Separates examples in a kernel feature space; similarity is \(K(x,x')=e^{-\gamma\lVert x-x'\rVert^2}\). |
| Random Forest | Handles nonlinear thresholds and interactions | Aggregates predictions from many bootstrapped decision trees. |
| AdaBoost | Tests sequential correction of hard examples | Adds weak learners while increasing emphasis on previous errors. |
| XGBoost | Strong gradient-boosted tree comparator | Sequentially fits trees to reduce a regularized classification loss. |

`compare_models.py` uses 200 estimators for the tree ensembles. Logistic Regression (`max_iter=3000`) and the RBF SVM use a pipeline that fits `StandardScaler` to numeric training columns and passes the binary indicators through. This matters because distance-based and coefficient-based methods are sensitive to feature scale. Tree models receive supplied numeric values directly because threshold splits generally do not need normalization. The optional `--smote` experiment uses training-only `SMOTENC` for the binary categorical indicators and a Random Forest; it is an auxiliary experiment, not the chosen deployment path.

## 5. Evaluation and observed results

The positive class is **Approved (1)**. The confusion matrix is `[[TN, FP], [FN, TP]]`: TN = rejected correctly; FP = rejected predicted approved; FN = approved predicted rejected; TP = approved correctly. The FP case is particularly relevant when discussing mistaken approvals.

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad
\text{Approved precision}=\frac{TP}{TP+FP},\quad
\text{Approved recall}=\frac{TP}{TP+FN},\quad
F_1=\frac{2PR}{P+R}.
\]

The same formulas are computed with Rejected as the positive class to report rejected precision, recall, and F1. ROC-AUC measures ranking across classification thresholds using predicted scores; it is not probability calibration.

| Feature set | Logistic Regression | RBF SVM | Random Forest | AdaBoost | XGBoost |
|---|---:|---:|---:|---:|---:|
| Original predictors | 92.39% | 94.85% | **98.36%** | 97.89% | 98.01% |
| Engineered predictors | 91.57% | 95.08% | **99.88%** | **99.88%** | **99.88%** |

These are accuracies on the supplied 854-row holdout. Engineered Random Forest and AdaBoost each yielded `[[322,1],[0,531]]`; XGBoost yielded `[[323,0],[1,530]]`. Thus the models tie in accuracy but make different types of single error. Logistic Regression declined slightly after adding ratios: engineering is not universally beneficial.

The separately saved **app model** is a Random Forest trained by `train_model.py` on matched, unscaled raw data plus three derived features. Its actual test accuracy is **853/854 = 99.8829%**, with `[[322,1],[0,531]]`, Approved precision **99.8120%**, Approved recall **100%**, Rejected recall **99.6904%**, and test ROC-AUC approximately **1.0**. Training accuracy is 100%. This is a separate fitted artifact from the processed-file comparison, even though its holdout accuracy and confusion counts match. We chose Random Forest because it was among the top performers, supports interactions, and is straightforward to use with raw-unit inputs; this choice does not establish that it is superior to the tied ensembles on new data.

Full results, including both classes' precision/recall/F1 and confusion counts, are in `results/model_comparison.csv` and `results/deployed_model_metrics.json`.

### Why is accuracy so high?

`check_data.py` found **zero exact predictor-row overlaps** across the supplied train and test partitions and no missing values. A depth-4 tree using only CIBIL achieved **96.37%** test accuracy; a depth-4 tree using all engineered features achieved **100%** on this same holdout. These observations suggest that the supplied labels are strongly separable by a small set of thresholds involving credit score, term, and financial ratios. They do **not** tell us how the source labels were generated, rule out subtler leakage, or predict performance on independent real applicants. The high score should be reported as **performance on this fixed dataset split**, not as real-world approval reliability. Multiple models were compared on the same holdout, so choosing the winner using that holdout can also make its quoted performance optimistic. The model's displayed probability is uncalibrated.

## 6. Split reconciliation and the earlier error

The initial version of training assumed the raw CSV rows appeared in exactly the same order as the supplied split. That assumption caused `train split differs from supplied files; check raw row order`. It was a row-order mismatch check, not a reason to ignore the supplied split.

The updated `train_model.py` joins each processed row to a raw row using a unique combination of six **input-derived** fields (`total_asset_value`, `dti`, `atr`, `education`, `self_employed`, `no_of_dependents`). The ratios are rounded to eight decimal places for matching. It enforces one-to-one uniqueness, confirms all 4,269 rows are assigned exactly once, compares target labels and three derived values, then checks that a training-fitted scaler reproduces the supplied original numerical columns. It fails with an explicit error when any check disagrees. The target is used only for validation and training, never as a matching key or predictor.

## 7. Application and code map

- `features.py`: input schema, category encoding, total assets, `dti`, and `atr`; shared by training and app.
- `check_data.py`: data quality, overlap, and shallow-tree audit; writes `results/data_audit.txt`.
- `compare_models.py`: five algorithms × two feature sets, optional SMOTENC; writes CSV and JSON comparison files.
- `train_model.py`: validates raw-to-processed membership, trains the raw-input Random Forest, writes metrics and `loan_approval_model.joblib`.
- `app.py`: Streamlit manual form, seven automatic demo scenarios, derived-value display, and prediction history.
- `test.py`: an earlier scenario runner; the current app's demo button works without Playwright or running this script.
- `results/`: recorded metrics and audit evidence.

The app loads its own trusted `loan_approval_model.joblib` and checks the feature order. Manual submissions and demo cases are saved with their input fields, derived values, predicted label, estimated Approved probability, and timestamp in local `history.sqlite3`. The history table is visible in the app. Seven built-in examples probe changes in CIBIL, assets, loan term, ratio, and categorical inputs; the listed “expected” labels are scenario expectations for demonstration, while the actual prediction is always computed by the saved model. The demo pauses two seconds between cases and writes each result to history.

## 8. Reproduce in VS Code (Windows)

From the `loan_approval_project` folder, open a terminal:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python check_data.py
python compare_models.py --smote
python train_model.py
python -m streamlit run app.py
```

Use PowerShell's `.\.venv\Scripts\Activate.ps1` in place of the Command Prompt activation command when appropriate. Place the original `data/loan_approval_dataset.csv` alongside the four processed CSVs before running `train_model.py`. Open the localhost address printed by Streamlit. Click **Start automated demo** to run all seven cases, or enter a manual profile and click **Predict**. Review saved cases under **Prediction history**. `test.py` and Playwright are not required for this flow.

For a GitHub repository, commit the source scripts, `requirements.txt`, this document, and reproducible evaluation outputs. Consider a `.gitignore` for `.venv/`, `__pycache__/`, `history.sqlite3`, and any local-only model or data artifacts. Check the source dataset's redistribution terms before publishing the CSV or `.joblib`.

## 9. Limits and next steps

The current evidence is one fixed holdout from one dataset, without a second genuinely independent or time-separated population. A production study would require provenance checks, external and temporal validation, subgroup/error analysis, probability calibration, monitoring, and lending-specific governance. The current app is suited to a team demonstration and methodological discussion.
