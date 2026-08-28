# Credit Risk PD/LGD/EAD Modelling — GitHub Repo Structure

## Folder Organization

```
credit-risk-modeling/
│
├── README.md                          # Project overview & results
├── requirements.txt                   # Dependencies
│
├── data/
│   └── loan_data.csv                  # Raw dataset (2,000 loans)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Rename: credit_risk.ipynb
│   ├── 02_feature_engineering.ipynb   # Rename: credit_risk2.ipynb
│   ├── 03_pd_model.ipynb              # Rename: PD_IFRS.ipynb
│   └── 04_vintage_analysis.ipynb      # Rename: vintage_analysis1.ipynb
│
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py         # Reusable feature functions
│   ├── model_training.py              # Model pipeline
│   └── evaluation.py                  # Metrics & evaluation
│
├── scripts/
│   └── credit_risk_pd_model.py        # Full pipeline (production version)
│
├── docs/
│   ├── Classification_Metrics_Explained.md  # Study notes (converted from docx)
│   ├── PD_LGD_EAD_Syllabus.md               # Structure/methodology
│   └── IFRS9_Basel_Framework.md            # Reference docs
│
└── results/
    ├── roc_curve.png                  # Model performance plots
    ├── feature_importance.png
    ├── vintage_analysis.png
    └── metrics_summary.txt            # Final AUC, F1, Recall, Precision
```

## What Goes Where

**notebooks/** → Keep your actual Jupyter notebooks (preprocessing, analysis, experimentation)

**src/** → Clean, reusable Python functions extracted from notebooks

**scripts/** → Full end-to-end pipeline script (the one I wrote)

**data/** → Your loan_data.csv

**docs/** → Study materials, methodology, explanations

**results/** → Outputs from models (plots, metrics)

## README.md Content Template

```markdown
# Credit Risk PD/LGD/EAD Modelling

Probability of Default (PD) prediction pipeline aligned with IFRS 9 & Basel II/III framework.

## Dataset
- **Source**: LendingClub loan data
- **Size**: 2,000 loans | 74 features
- **Target**: Loan status (Default vs Fully Paid)
- **Default Rate**: 40.5%

## Model Architecture
- **Algorithms**: Logistic Regression, Gradient Boosting
- **Features Engineered**: 15 domain-specific features
- **Evaluation**: AUC-ROC, F1-score, Precision, Recall, ROC Curve

## Key Results
- Gradient Boosting F1-score: 0.28
- Precision: 0.49 | Recall: 0.19
- Vintage Analysis: Cohort-level default tracking

## Project Structure
- `/notebooks/` — Exploratory analysis & feature engineering
- `/src/` — Reusable pipeline functions
- `/scripts/` — Full end-to-end model training
- `/data/` — Raw dataset
- `/docs/` — Methodology & IFRS 9 references

## How to Run
```bash
python scripts/credit_risk_pd_model.py
```

## Methodology
- Binary classification: Default vs Non-Default
- Feature engineering: Credit history, employment, recovery rates
- Train-test: 80/20 stratified split
- Vintage Analysis: Cohort default tracking over time
- Risk Framework: IFRS 9 provisioning, Basel III expected loss

## Resume Bullet
[Insert the strong bullet I wrote above]
```

