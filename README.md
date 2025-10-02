[README.md](https://github.com/user-attachments/files/22671107/README.md)
# Alzheimer / Cognitive Impairment Classification — README

> **Project:** Exploratory data analysis & baseline classifiers to predict cognitive impairment (HC vs NON-HC) from merged AIBL CSVs.  
> **Goal:** Clean, visualize and train baseline models (LightGBM / RandomForest) to separate Healthy Controls (HC) from Non-Healthy (MCI + AD).

---

## Table of contents
- [Project overview](#project-overview)  
- [Files & dataset](#files--dataset)  
- [Summary of data processing](#summary-of-data-processing)  
- [Exploratory Data Analysis (EDA) highlights](#exploratory-data-analysis-eda-highlights)  
- [Feature selection](#feature-selection)  
- [Modeling & evaluation](#modeling--evaluation)  
- [How to run (Colab / local)](#how-to-run-colab--local)  
- [Dependencies](#dependencies)  
- [Next steps & improvements](#next-steps--improvements)  
- [License & contact](#license--contact)

---

## Project overview
This notebook merges multiple AIBL CSVs (APOE, CDR, lab data, medical history, MMSE, neurobat, pdxconv, patient demographics) into a single dataframe and performs:

- cleaning & basic transformations (age calculation, target encoding),
- exploratory visualizations (distributions, KDEs, countplots),
- simple feature selection (reducing to 17 predictive features),
- baseline training using LightGBM / RandomForest,
- evaluation with metrics like ROC AUC and classification reports.

The final target is a binary label: `0 = HC` (Healthy Control), `1 = NON HC` (MCI or AD).

---

## Files & dataset
Expect the following CSVs (used in the notebook):

```
/content/aibl_apoeres_01-Jun-2018.csv
/content/aibl_cdr_01-Jun-2018.csv
/content/aibl_labdata_01-Jun-2018.csv
/content/aibl_medhist_01-Jun-2018.csv
/content/aibl_mmse_01-Jun-2018.csv
/content/aibl_neurobat_01-Jun-2018.csv
/content/aibl_pdxconv_01-Jun-2018.csv
/content/aibl_ptdemog_01-Jun-2018.csv
```

Notebook output: merged `df` (shape printed as `(1688, 36)` in example); selected features subset saved to `features_selected`.

> **Note:** Keep CSVs in the same folder (Colab: `/content/`) or change the paths accordingly.

---

## Summary of data processing

1. **Merging:** multiple tables merged on `['RID','SITEID','VISCODE']` or `RID` depending on source.
2. **Date → Age:** `PTDOB` was parsed and used to compute `age = 2021 - birth_year` (adapt year if you re-run).
3. **Target mapping:** original `DXCURREN` values converted:
   - dropped `-4` and `7` (invalid / unclear),
   - mapped `1 → "HC"`, else `"NON HC"`, then binary `HC → 0`, `NON HC → 1`.
4. **Dropped columns:** `EXAMDATE_x`, `EXAMDATE_y`, `SITEID`, `VISCODE` (kept where useful).
5. **Missing / sentinel values:** dataset contains `-4` as a sentinel for missing — many visualizations filtered those rows for the specific variable before plotting (e.g., `df[df.MH16SMOK != -4]`).

---

## EDA highlights

- **Class imbalance:** ~73% HC, 27% NON HC. (You used a bar plot of normalized counts.)
- **MMSCORE:** Healthy controls score higher on MMSE than MCI/AD groups (kde showed separation).
- **Age:** More concentration of older ages (90–100) in the MCI/AD group in your sample.
- **Smoking, psych history, malignancy:** Crosstabs suggested limited association in this dataset (after filtering `-4`).
- **Memory tests:** `LIMMTOTAL` and `LDELTOTAL` show HC > NON HC on immediate and delayed recall.

---

## Feature selection

You reduced the feature pool from ~30 to **17** features (those with best predictive power in your experiments):

```
['RCT11','HMT40','RCT6','HMT13','MH9ENDO','LIMMTOTAL','MMSCORE','AXT117',
 'RCT392','HMT100','HMT7','age','CDGLOBAL','BAT126','HMT102','LDELTOTAL','RCT20']
```

These are stored in `features_selected`.

---

## Modeling & evaluation (baseline training)

The provided `train.py` script trains a LightGBM classifier on the selected features and evaluates using ROC AUC and classification report.

**Run:**
```bash
python train.py
```

It will save the model as `lgbm_baseline.pkl`.

---

## How to run (Colab / local)

1. **In Colab**
   - Upload the CSV files to `/content/` or mount Google Drive.
   - Create a new notebook and paste your code cells.
   - Install missing packages if needed:
     ```bash
     !pip install -r requirements.txt
     ```

2. **Locally**
   - Create a virtual environment and install requirements:
     ```bash
     pip install -r requirements.txt
     ```
   - Place CSVs in a `data/` folder and update file paths in the notebook or preprocessing script.
   - Run:
     ```bash
     python train.py
     ```

---

## Requirements

See [requirements.txt](requirements.txt):

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
```

---

## Next steps & improvements
- Hyperparameter tuning (GridSearch / Optuna).
- Feature engineering (interactions, scaling, categorical encoding).
- More principled imputation (KNN, IterativeImputer).
- Class imbalance handling (SMOTE, ADASYN, class-weighted losses).
- Explainability with SHAP.
- Longitudinal modeling with VISCODE/EXAMDATE.
- Full stratified cross-validation.

---

## License & contact
- Suggested license: **MIT License** (or your organization’s preferred license).
- For questions / help improving the notebook: **Akinyera Akintunde** (use your preferred contact details).
