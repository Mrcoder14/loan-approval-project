# Loan approval classification: reproducible comparison and UI

The four CSVs in `data/` are the user's **already processed** train/test splits. `original` means original predictor columns, **not raw numeric values**: income, loan amount, term, CIBIL and asset columns have already been standardized using the training split. The `engineered` files include these scaled columns plus raw `total_asset_value`, `dti` and `atr`.

## Windows VS Code instructions

1. Extract the ZIP. Open its `loan_approval_project` folder in VS Code. Open **Terminal → New Terminal** and use PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   If PowerShell refuses the activation script, use Command Prompt instead: `.venv\Scripts\activate.bat`. Select the `.venv` interpreter in VS Code.

2. Run the five algorithms on both supplied splits. The comparison uses the **same fixed test set** for every model and fits scaling for Logistic Regression and SVM on the training rows only. No sampler touches the test set.

   ```powershell
   python check_data.py
   python compare_models.py --smote
   ```

   Read `results/model_comparison.csv` for train/test accuracy, precision, recall, F1, ROC-AUC, and all four confusion-matrix counts. The optional comparison uses **SMOTENC** for the two encoded binary columns, applied to training data only; the unsampled models are the primary comparison.

3. **To train the interactive app on human-readable inputs:** download Archit Sharma's [original Kaggle dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) and put `loan_approval_dataset.csv` inside `data/` (beside the four supplied CSVs). The raw CSV was not among the four attachments. Do not rename `train_original.csv` to this: it contains scaled data.

   ```powershell
   python train_model.py
   python -m streamlit run app.py
   ```

   `train_model.py` verifies that the raw CSV recreates the supplied train/test labels and three engineered columns before saving `loan_approval_model.joblib`; a wrong dataset or row order fails loudly. The browser usually opens at `http://localhost:8501`. Stop it with Ctrl+C. See `results/deployed_model_metrics.json` for the **actual saved model** train/test metrics. These may differ from the processed-CSV benchmark.

## What the metrics mean

`Approved=1`; confusion matrix `[[TN, FP], [FN, TP]]` with `Rejected=0`. Here `FP` means a rejected label predicted Approved; `FN` means an approved label predicted Rejected. Both class-specific precision, recall and F1 are shown. ROC-AUC uses the model's probability score. `1.0000` ROC-AUC may be rounded; use the full CSV value. The Random Forest in `train_model.py` uses raw numerical features, engineered features, and **no SMOTE or StandardScaler**; its saved metrics are printed separately.

The benchmark has very high held-out results on engineered features. `check_data.py` checks exact duplicate predictor rows across splits and shows that a depth-four decision tree can separate the provided held-out labels. CIBIL alone is highly predictive. That pattern merits caution about dataset provenance and generalization; exact train/test overlap was not found, and the derived columns are made from predictor values rather than from the target. This is an academic classifier of dataset approval labels, **not** a validated default-risk model or actual lending decision. Probabilities are not calibrated.

## Project files

- `compare_models.py`: five-model, two-feature-set comparison; optional RF with SMOTENC.
- `check_data.py`: data and simple-rule audit.
- `features.py`: shared raw input transformations for training and UI.
- `train_model.py`: deployable Random Forest trained from verified raw CSV.
- `app.py`: Streamlit form using saved model and same feature order.
- `results/`: outputs from the run on the supplied splits.

Never put `loan_status` or `loan_id` into the model inputs. Keep the raw data and generated `.joblib` out of public repositories if you do not have rights to distribute them. A serialized model should only be loaded from your own trusted run.
