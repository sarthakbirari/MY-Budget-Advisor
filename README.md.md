# Student Budget Advisor

---

<!-- ============================================================ -->
<!-- ============================================================ -->

**Student Name:** Sarthak Birari

**Roll Number:** 25MIM10046

**Course:** Fundamentals of AI and ML

**Project:** BYOP – Student Budget Advisor

**Platform:** VITyarthi

<!-- ============================================================ -->

---

> A machine learning project that predicts whether a student will overshoot their monthly budget and recommends a concrete spending reallocation to help them stay within limits.

---

## What This Project Does

Managing a monthly budget is a real challenge for most students, especially those living away from home. This project uses supervised ML classification to:

1. **Predict** — given a student's allowance and planned spending, will they overshoot their budget?
2. **Advise** — if overshoot is predicted, which categories should they cut and by how much?

Three models are trained and compared: Logistic Regression, Decision Tree, and Random Forest. The best model (selected by AUC-ROC score) is then used in a rule-based advisor layer that outputs specific, actionable recommendations.

---

## Project Structure

```
student-budget-advisor/
│
├── budget_advisor.py         # Main ML pipeline (run this)
├── student_budget_data.csv   # Auto-generated synthetic dataset (800 records)
│
├── eda_plots.png             # EDA visualisations (auto-generated)
├── model_evaluation.png      # Confusion matrix + ROC curves (auto-generated)
├── feature_importance.png    # Random Forest feature importance (auto-generated)
│
└── README.md
```

---

## Features Used

| Feature | Description |
|---|---|
| `allowance` | Monthly income/allowance in Rs. |
| `rent` | Monthly rent paid |
| `food` | Food and dining expenses |
| `transport` | Commute, fuel, ride-sharing |
| `entertainment` | OTT, outings, leisure |
| `study_material` | Books, stationery, courses |
| `miscellaneous` | Everything else |
| `spend_to_income` | Total spend divided by allowance (key feature) |
| `savings_rate` | (Allowance minus spend) divided by allowance |
| `discretionary` | entertainment + miscellaneous |
| `essential` | rent + food + transport + study_material |

**Target:** `overshoot` — `1` if total spend exceeds allowance, `0` otherwise.

---

## Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
git clone https://github.com/sarthakbirari/MY-Budget-Advisor.git
cd student-budget-advisor

# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn
```

No GPU required. Runs on any laptop.

---

## How to Run

```bash
python budget_advisor.py
```

This single command will:

1. Generate a synthetic dataset of 800 student records
2. Run exploratory data analysis and save plots
3. Train and compare three ML models
4. Print a full classification report for the best model
5. Run two demo scenarios showing the advisor in action

---

## Sample Output

```
============================================================
  STUDENT BUDGET ADVISOR - ML Pipeline
============================================================
  Student : Sarthak Birari
  Roll No : 25MIM10046
  Course  : Fundamentals of AI and ML
============================================================

[4] Model Comparison
--------------------------------------------------
  Logistic Regression       Acc: 0.694  AUC: 0.764  CV-Acc: 0.727
  Decision Tree             Acc: 0.656  AUC: 0.643  CV-Acc: 0.706
  Random Forest             Acc: 0.706  AUC: 0.726  CV-Acc: 0.717

Scenario A - Tight Budget Student
  Allowance  : Rs.12,000
  Total Spend: Rs.14,500
  Prediction : OVERSHOOT LIKELY
  Probability: 70.8%
  Advice:
  * Reduce entertainment by Rs.750/month
  * Reduce miscellaneous by Rs.375/month
  * Cook at home to save ~Rs.500/month

Scenario B - Comfortable Budget Student
  Allowance  : Rs.18,000
  Total Spend: Rs.10,300
  Prediction : WITHIN BUDGET
  Probability: 19.0%
  Advice:
  * Great! You have a surplus of Rs.7700.
  * Consider saving Rs.3850 this month.
```

---

## Model Performance Summary

| Model | Accuracy | AUC-ROC | 5-Fold CV Acc |
|---|---|---|---|
| Logistic Regression | 69.4% | **0.764** (Best) | 72.7% |
| Decision Tree | 65.6% | 0.643 | 70.6% |
| Random Forest | 70.6% | 0.726 | 71.7% |

Logistic Regression was selected as the best model based on AUC score.

---

## Concepts Applied

- Binary classification with scikit-learn
- Synthetic dataset generation with NumPy
- Exploratory data analysis with Matplotlib and Seaborn
- Feature engineering (derived financial ratios)
- StandardScaler for feature normalisation
- Model comparison: Logistic Regression vs Decision Tree vs Random Forest
- Cross-validation and AUC-ROC evaluation
- Rule-based advisor layer on top of ML predictions

---

## License

MIT License — free to use, modify, and distribute with attribution.
