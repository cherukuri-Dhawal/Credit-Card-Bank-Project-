"""
Credit Risk PD Model — LendingClub Full Dataset
================================================
Dataset : https://www.kaggle.com/datasets/wordsforthewise/lending-club
File    : accepted_2007_to_2018Q4.csv.gz  (~2.2M rows, 150 cols)
Run     : python credit_risk_pd_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, classification_report, roc_curve,
                             confusion_matrix)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")

FILE_PATH = "accepted_2007_to_2018Q4.csv.gz"   # <-- change if needed

# Load in chunks for memory efficiency — only columns we need
COLS = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'grade',
    'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'loan_status', 'purpose', 'dti',
    'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_pymnt',
    'total_rec_prncp', 'total_rec_int', 'recoveries', 'last_pymnt_amnt'
]

df = pd.read_csv(FILE_PATH, usecols=COLS, low_memory=False)
print(f"Raw shape: {df.shape}")
print(f"Loan status distribution:\n{df['loan_status'].value_counts()}\n")


# ─────────────────────────────────────────────
# STEP 2: DEFINE DEFAULT (PD Target Variable)
# ─────────────────────────────────────────────
# Only keep rows with clear outcome — remove ambiguous statuses
GOOD = ['Fully Paid']
BAD  = ['Charged Off', 'Default']

df = df[df['loan_status'].isin(GOOD + BAD)].copy()
df['default'] = df['loan_status'].apply(lambda x: 1 if x in BAD else 0)

print(f"After filtering — rows: {len(df)}")
print(f"Default rate: {df['default'].mean()*100:.1f}%\n")


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Employment length → integer
df['emp_length_int'] = df['emp_length'].str.extract(r'(\d+)').astype(float)

# Credit history age in months
df['earliest_cr_line_date'] = pd.to_datetime(
    df['earliest_cr_line'], format='%b-%Y', errors='coerce'
)
df.loc[df['earliest_cr_line_date'].dt.year > 2025,
       'earliest_cr_line_date'] -= pd.DateOffset(years=100)

REFERENCE_DATE = pd.to_datetime('2019-01-01')
df['credit_history_months'] = (
    (REFERENCE_DATE - df['earliest_cr_line_date']).dt.days / 30.44
).round()

# Grade → ordinal
grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
df['grade_int'] = df['grade'].map(grade_map)

# Home ownership → binary flags
df['home_ownership_own']  = (df['home_ownership'] == 'OWN').astype(int)
df['home_ownership_rent'] = (df['home_ownership'] == 'RENT').astype(int)

# Recovery rate proxy (LGD input)
df['recovery_rate'] = df['recoveries'] / (df['loan_amnt'] + 1)

# int_rate — strip % if string
df['int_rate'] = pd.to_numeric(
    df['int_rate'].astype(str).str.replace('%',''), errors='coerce'
)
df['revol_util'] = pd.to_numeric(
    df['revol_util'].astype(str).str.replace('%',''), errors='coerce'
)

print("Feature engineering done.\n")


# ─────────────────────────────────────────────
# STEP 4: SELECT FEATURES & CLEAN
# ─────────────────────────────────────────────
FEATURES = [
    'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'last_pymnt_amnt',
    'emp_length_int', 'credit_history_months', 'grade_int',
    'home_ownership_own', 'home_ownership_rent', 'recovery_rate'
]

df_model = df[FEATURES + ['default']].dropna()
print(f"Model-ready rows: {len(df_model)}")
print(f"Features used   : {len(FEATURES)}\n")

X = df_model[FEATURES]
y = df_model['default']


# ─────────────────────────────────────────────
# STEP 5: TRAIN / TEST SPLIT (Stratified 80/20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")


# ─────────────────────────────────────────────
# STEP 6: LOGISTIC REGRESSION (PD Model)
# ─────────────────────────────────────────────
print("Training Logistic Regression...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
lr.fit(X_train_s, y_train)

lr_probs = lr.predict_proba(X_test_s)[:, 1]
lr_preds = lr.predict(X_test_s)

lr_auc = roc_auc_score(y_test, lr_probs)
lr_f1  = f1_score(y_test, lr_preds)
lr_pre = precision_score(y_test, lr_preds)
lr_rec = recall_score(y_test, lr_preds)

print(f"Logistic Regression → AUC: {lr_auc:.4f} | F1: {lr_f1:.4f} | "
      f"Precision: {lr_pre:.4f} | Recall: {lr_rec:.4f}")


# ─────────────────────────────────────────────
# STEP 7: GRADIENT BOOSTING (Better PD Model)
# ─────────────────────────────────────────────
print("\nTraining Gradient Boosting (this takes ~2-3 mins)...")
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)

gb_probs = gb.predict_proba(X_test)[:, 1]
gb_preds = gb.predict(X_test)

gb_auc = roc_auc_score(y_test, gb_probs)
gb_f1  = f1_score(y_test, gb_preds)
gb_pre = precision_score(y_test, gb_preds)
gb_rec = recall_score(y_test, gb_preds)

print(f"Gradient Boosting   → AUC: {gb_auc:.4f} | F1: {gb_f1:.4f} | "
      f"Precision: {gb_pre:.4f} | Recall: {gb_rec:.4f}")

print("\n--- Full Classification Report (Gradient Boosting) ---")
print(classification_report(y_test, gb_preds))


# ─────────────────────────────────────────────
# STEP 8: CROSS VALIDATION (5-Fold)
# ─────────────────────────────────────────────
print("\nRunning 5-Fold Cross Validation on Gradient Boosting...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"CV AUC scores : {cv_scores.round(4)}")
print(f"Mean CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─────────────────────────────────────────────
# STEP 9: ROC CURVE PLOT
# ─────────────────────────────────────────────
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_auc:.3f})', linewidth=2)
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC={gb_auc:.3f})', linewidth=2)
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Credit Risk PD Model (LendingClub)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print("\nROC curve saved as roc_curve.png")


# ─────────────────────────────────────────────
# STEP 10: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = pd.Series(gb.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("\n--- Top 10 Feature Importances ---")
print(importances.head(10).round(4))

plt.figure(figsize=(10, 6))
importances.head(10).plot(kind='barh', color='steelblue')
plt.title('Top 10 Feature Importances — Gradient Boosting PD Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Feature importance plot saved as feature_importance.png")


# ─────────────────────────────────────────────
# STEP 11: VINTAGE ANALYSIS
# ─────────────────────────────────────────────
print("\n--- Vintage Analysis ---")
df_vintage = df[['loan_status', 'earliest_cr_line_date', 'default']].dropna()
df_vintage['vintage_year'] = df_vintage['earliest_cr_line_date'].dt.year
vintage = df_vintage.groupby('vintage_year')['default'].agg(['mean','count'])
vintage.columns = ['default_rate', 'loan_count']
vintage = vintage[vintage['loan_count'] > 100]
print(vintage.round(4))

plt.figure(figsize=(10, 5))
plt.plot(vintage.index, vintage['default_rate']*100, marker='o', linewidth=2, color='crimson')
plt.title('Vintage Analysis — Default Rate by Credit History Year')
plt.xlabel('Vintage Year')
plt.ylabel('Default Rate (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vintage_analysis.png', dpi=150)
plt.show()
print("Vintage analysis plot saved as vintage_analysis.png")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL RESULTS SUMMARY")
print("="*55)
print(f"Dataset          : LendingClub (~2.2M loans, {len(FEATURES)} features)")
print(f"Model-ready rows : {len(df_model):,}")
print(f"Default rate     : {y.mean()*100:.1f}%")
print(f"")
print(f"Logistic Regression → AUC: {lr_auc:.4f} | F1: {lr_f1:.4f}")
print(f"Gradient Boosting   → AUC: {gb_auc:.4f} | F1: {gb_f1:.4f}")
print(f"5-Fold CV AUC       → {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("="*55)
print("\nCopy the metrics above for your resume bullet!")
