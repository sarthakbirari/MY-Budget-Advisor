"""
============================================================
============================================================
"""
STUDENT_NAME = "Sarthak Birari"       
ROLL_NUMBER  = "25MIM10046"    
COURSE       = "Fundamentals of AI and ML"
PROJECT      = "BYOP - Student Budget Advisor"
"""
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

np.random.seed(42)
N = 800

def generate_dataset(n):
    allowance      = np.random.randint(5000, 25001, n)
    rent           = np.random.randint(1500, 8001, n)
    food           = np.random.randint(800,  5001, n)
    transport      = np.random.randint(200,  2001, n)
    entertainment  = np.random.randint(0,    3001, n)
    study_material = np.random.randint(0,    1501, n)
    miscellaneous  = np.random.randint(0,    2001, n)

    total_spend     = rent + food + transport + entertainment + study_material + miscellaneous
    savings_rate    = (allowance - total_spend) / allowance
    spend_to_income = total_spend / allowance
    discretionary   = entertainment + miscellaneous
    essential       = rent + food + transport + study_material

    overshoot_prob = 1 / (1 + np.exp(-3 * (spend_to_income - 0.95)))
    noise          = np.random.normal(0, 0.05, n)
    overshoot_prob = np.clip(overshoot_prob + noise, 0, 1)
    overshoot      = (np.random.rand(n) < overshoot_prob).astype(int)

    return pd.DataFrame({
        "allowance": allowance, "rent": rent, "food": food,
        "transport": transport, "entertainment": entertainment,
        "study_material": study_material, "miscellaneous": miscellaneous,
        "total_spend": total_spend, "savings_rate": savings_rate.round(4),
        "spend_to_income": spend_to_income.round(4),
        "discretionary": discretionary, "essential": essential,
        "overshoot": overshoot
    })

df = generate_dataset(N)
df.to_csv("student_budget_data.csv", index=False)

print("=" * 60)
print("  STUDENT BUDGET ADVISOR - ML Pipeline")
print("=" * 60)
print(f"  Student : {STUDENT_NAME}")
print(f"  Roll No : {ROLL_NUMBER}")
print(f"  Course  : {COURSE}")
print("=" * 60)
print(f"\n[1] Dataset generated: {N} student records")
print(f"    Overshoot rate: {df['overshoot'].mean():.1%}")


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(f"Student Budget Advisor - EDA\n{STUDENT_NAME} | {ROLL_NUMBER}",
             fontsize=14, fontweight="bold")

axes[0, 0].bar(["Within Budget", "Overshoot"],
               df["overshoot"].value_counts().sort_index(),
               color=["#4CAF50", "#E53935"])
axes[0, 0].set_title("Budget Overshoot Distribution")
axes[0, 0].set_ylabel("Count")

for label, colour in zip([0, 1], ["#4CAF50", "#E53935"]):
    subset = df[df["overshoot"] == label]["spend_to_income"]
    axes[0, 1].hist(subset, bins=30, alpha=0.6, color=colour,
                    label=["Within Budget", "Overshoot"][label])
axes[0, 1].set_title("Spend-to-Income Ratio by Class")
axes[0, 1].set_xlabel("Spend / Allowance")
axes[0, 1].legend()

cats = ["rent", "food", "transport", "entertainment", "study_material", "miscellaneous"]
axes[0, 2].barh(cats, df[cats].mean(), color="#1565C0")
axes[0, 2].set_title("Average Monthly Spend by Category")
axes[0, 2].set_xlabel("Rs.")

colours = df["overshoot"].map({0: "#4CAF50", 1: "#E53935"})
axes[1, 0].scatter(df["allowance"], df["total_spend"], c=colours, alpha=0.4, s=15)
axes[1, 0].plot([5000, 25000], [5000, 25000], "k--", lw=1, label="Break-even")
axes[1, 0].set_title("Allowance vs Total Spend")
axes[1, 0].set_xlabel("Allowance (Rs.)")
axes[1, 0].set_ylabel("Total Spend (Rs.)")
axes[1, 0].legend()

corr = df[cats + ["total_spend", "spend_to_income", "overshoot"]].corr()
sns.heatmap(corr, ax=axes[1, 1], cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5, annot_kws={"size": 7})
axes[1, 1].set_title("Feature Correlation Heatmap")

for label, colour, name in zip([0, 1], ["#4CAF50", "#E53935"], ["Within Budget", "Overshoot"]):
    sub = df[df["overshoot"] == label]
    axes[1, 2].scatter(sub["essential"], sub["discretionary"],
                       c=colour, alpha=0.4, s=15, label=name)
axes[1, 2].set_title("Essential vs Discretionary Spend")
axes[1, 2].set_xlabel("Essential Spend (Rs.)")
axes[1, 2].set_ylabel("Discretionary Spend (Rs.)")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[2] EDA plots saved -> eda_plots.png")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

FEATURES = [
    "allowance", "rent", "food", "transport",
    "entertainment", "study_material", "miscellaneous",
    "spend_to_income", "savings_rate", "discretionary", "essential"
]
TARGET = "overshoot"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n[3] Train size: {len(X_train)} | Test size: {len(X_test)}")


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
}

results = {}
print("\n[4] Model Comparison")
print("-" * 50)

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(
        model,
        X_train_sc if name == "Logistic Regression" else X_train,
        y_train, cv=5, scoring="accuracy"
    ).mean()

    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob,
                     "accuracy": acc, "auc": auc, "cv_acc": cv}
    print(f"  {name:<25} Acc: {acc:.3f}  AUC: {auc:.3f}  CV-Acc: {cv:.3f}")


# ─────────────────────────────────────────────
# 5. BEST MODEL EVALUATION
# ─────────────────────────────────────────────

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]

print(f"\n[5] Best Model -> {best_name} (AUC: {best['auc']:.3f})")
print("\nClassification Report:")
print(classification_report(y_test, best["y_pred"],
                             target_names=["Within Budget", "Overshoot"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Model Evaluation - {best_name}\n{STUDENT_NAME} | {ROLL_NUMBER}",
             fontsize=13, fontweight="bold")

cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Within Budget", "Overshoot"],
            yticklabels=["Within Budget", "Overshoot"])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.2f})")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves - All Models")
axes[1].legend(fontsize=8)

names = list(results.keys())
accs  = [results[n]["accuracy"] for n in names]
bars  = axes[2].barh(names, accs, color=["#1565C0", "#43A047", "#FB8C00"])
axes[2].set_xlim(0.5, 1.0)
axes[2].set_xlabel("Test Accuracy")
axes[2].set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accs):
    axes[2].text(acc + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Evaluation plots saved -> model_evaluation.png")


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

rf_model    = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()

plt.figure(figsize=(9, 6))
importances.plot(kind="barh", color="#1565C0")
plt.title(f"Feature Importance - Random Forest\n{STUDENT_NAME} | {ROLL_NUMBER}",
          fontsize=13, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Feature importance saved -> feature_importance.png")


# ─────────────────────────────────────────────
# 7. BUDGET REALLOCATION ADVISOR
# ─────────────────────────────────────────────

def advise_student(allowance, rent, food, transport,
                   entertainment, study_material, miscellaneous):
    total_spend    = rent + food + transport + entertainment + study_material + miscellaneous
    savings_rate   = (allowance - total_spend) / allowance
    spend_ratio    = total_spend / allowance
    discretionary  = entertainment + miscellaneous
    essential      = rent + food + transport + study_material

    sample = pd.DataFrame([{
        "allowance": allowance, "rent": rent, "food": food,
        "transport": transport, "entertainment": entertainment,
        "study_material": study_material, "miscellaneous": miscellaneous,
        "spend_to_income": spend_ratio, "savings_rate": savings_rate,
        "discretionary": discretionary, "essential": essential,
    }])

    model        = results[best_name]["model"]
    sample_input = scaler.transform(sample[FEATURES]) if best_name == "Logistic Regression" \
                   else sample[FEATURES]

    prediction = model.predict(sample_input)[0]
    confidence = model.predict_proba(sample_input)[0][1]

    advice = []
    if prediction == 1:
        deficit = total_spend - allowance
        if entertainment > 500:
            cut = min(entertainment * 0.30, deficit * 0.5)
            advice.append(f"  * Reduce entertainment by Rs.{cut:.0f}/month")
            deficit -= cut
        if miscellaneous > 300 and deficit > 0:
            cut = min(miscellaneous * 0.25, deficit)
            advice.append(f"  * Reduce miscellaneous by Rs.{cut:.0f}/month")
            deficit -= cut
        if food > 2000 and deficit > 0:
            advice.append(f"  * Cook at home to save ~Rs.{min(500, deficit):.0f}/month")
        if transport > 800 and deficit > 0:
            advice.append(f"  * Use a monthly transit pass (~Rs.{min(200, deficit):.0f} savings)")
        if not advice:
            advice.append("  * Review all spending categories carefully.")
    else:
        surplus = allowance - total_spend
        advice.append(f"  * Great! You have a surplus of Rs.{surplus:.0f}.")
        advice.append(f"  * Consider saving Rs.{surplus * 0.5:.0f} this month.")

    return {
        "prediction": "OVERSHOOT LIKELY" if prediction == 1 else "WITHIN BUDGET",
        "overshoot_probability": f"{confidence:.1%}",
        "total_spend": total_spend,
        "allowance": allowance,
        "advice": advice,
    }


# ── Demo ─────────────────────────────────────
print("\n[6] Budget Advisor Demo")
print("=" * 50)

result_a = advise_student(12000, 5000, 3500, 1200, 2500, 800, 1500)
print("\nScenario A - Tight Budget Student")
print(f"  Allowance  : Rs.{result_a['allowance']:,}")
print(f"  Total Spend: Rs.{result_a['total_spend']:,}")
print(f"  Prediction : {result_a['prediction']}")
print(f"  Probability: {result_a['overshoot_probability']}")
print("  Advice:")
for line in result_a["advice"]: print(line)

result_b = advise_student(18000, 5000, 2500, 800, 1000, 500, 500)
print("\nScenario B - Comfortable Budget Student")
print(f"  Allowance  : Rs.{result_b['allowance']:,}")
print(f"  Total Spend: Rs.{result_b['total_spend']:,}")
print(f"  Prediction : {result_b['prediction']}")
print(f"  Probability: {result_b['overshoot_probability']}")
print("  Advice:")
for line in result_b["advice"]: print(line)

print(f"\n[Done] All outputs saved.")
print(f"Submitted by: {STUDENT_NAME} | {ROLL_NUMBER}")
