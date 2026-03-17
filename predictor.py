"""
Student Enrolment Predictor
============================
Uses linear regression to predict future student enrolments
based on historical marketing spend and agent count data.

Full ML pipeline: data -> features -> model -> predictions -> visualisation.

Author: Pradeep Pandey
Tools: Scikit-learn, Pandas, Matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# -- 1. Historical Data -------------------------------------------------------
# Simulated 3 years of monthly data for an EdTech company
np.random.seed(42)
n_months = 36

data = {
    "month": range(1, n_months + 1),
    "marketing_spend_lakhs": np.round(
        np.linspace(2.0, 8.0, n_months) + np.random.normal(0, 0.3, n_months), 2
    ),
    "num_agents": np.round(
        np.linspace(10, 30, n_months) + np.random.normal(0, 1.5, n_months)
    ).astype(int),
    "avg_course_fee_thousands": np.round(
        np.linspace(45, 65, n_months) + np.random.normal(0, 2, n_months), 1
    ),
}

# Target: enrolments are influenced by all three features
data["students_enrolled"] = np.round(
    20 * data["marketing_spend_lakhs"]
    + 8 * data["num_agents"]
    - 0.5 * data["avg_course_fee_thousands"]
    + np.random.normal(0, 15, n_months)
).astype(int)

df = pd.DataFrame(data)

print("=" * 55)
print("       STUDENT ENROLMENT PREDICTOR")
print("=" * 55)

print("\n Dataset Overview:")
print(f"   Months of data   : {len(df)}")
print(f"   Features used    : Marketing Spend, No. of Agents, Avg Fee")
print(f"   Target variable  : Students Enrolled")
print(f"\n   Enrolment range  : {df['students_enrolled'].min()} - {df['students_enrolled'].max()}")
print(f"   Average monthly  : {df['students_enrolled'].mean():.0f} students")

# -- 2. Prepare Features & Target ---------------------------------------------
features = ["marketing_spend_lakhs", "num_agents", "avg_course_fee_thousands"]
X = df[features]
y = df["students_enrolled"]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -- 3. Train Linear Regression Model -----------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -- 4. Evaluate Model ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n Model Performance:")
print(f"   Algorithm        : Linear Regression")
print(f"   Training samples : {len(X_train)} months")
print(f"   Test samples     : {len(X_test)} months")
print(f"   R2 Score         : {r2:.3f}  (1.0 = perfect, >0.8 = good)")
print(f"   Mean Abs Error   : {mae:.1f} students")

print(f"\n Feature Coefficients (impact on enrolments):")
for feature, coef in zip(features, model.coef_):
    direction = "up" if coef > 0 else "down"
    print(f"   {feature:<35} {direction} {abs(coef):.1f} students per unit")

# -- 5. Predict Next 3 Months -------------------------------------------------
future_scenarios = pd.DataFrame({
    "marketing_spend_lakhs":     [8.5,  9.0,  9.5],
    "num_agents":                [31,   33,   35],
    "avg_course_fee_thousands":  [66.0, 67.0, 68.0],
})

future_scaled = scaler.transform(future_scenarios)
future_preds = model.predict(future_scaled).astype(int)

print(f"\n Forecast - Next 3 Months:")
print(f"   {'Month':<10} {'Mktg Spend':>12} {'Agents':>8} {'Predicted Enrolments':>22}")
print(f"   {'-'*56}")
for i, (_, row) in enumerate(future_scenarios.iterrows()):
    print(f"   Month {n_months + i + 1:<4}  {row['marketing_spend_lakhs']:.1f}L        "
          f"{row['num_agents']:<8} {future_preds[i]:>20,} students")

# -- 6. Visualisations ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Student Enrolment Predictor", fontsize=15, fontweight="bold")

# Chart 1 - Actual vs Predicted (test set)
axes[0].scatter(y_test, y_pred, color="#2196F3", alpha=0.8, edgecolors="white", s=60)
min_val = min(y_test.min(), y_pred.min()) - 20
max_val = max(y_test.max(), y_pred.max()) + 20
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")
axes[0].set_xlabel("Actual Enrolments")
axes[0].set_ylabel("Predicted Enrolments")
axes[0].set_title(f"Actual vs Predicted (R2 = {r2:.3f})")
axes[0].legend()

# Chart 2 - Actual enrolments over time + forecast
axes[1].plot(df["month"], df["students_enrolled"], color="#4CAF50",
             linewidth=2, label="Historical", marker="o", markersize=3)
forecast_months = [n_months + 1, n_months + 2, n_months + 3]
axes[1].plot(forecast_months, future_preds, color="#FF9800",
             linewidth=2, linestyle="--", label="Forecast", marker="s", markersize=6)
axes[1].axvline(n_months, color="grey", linestyle=":", linewidth=1)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Students Enrolled")
axes[1].set_title("Enrolment Trend + 3-Month Forecast")
axes[1].legend()

plt.tight_layout()
plt.savefig("enrolment_prediction_chart.png", dpi=150, bbox_inches="tight")
print(f"\nChart saved to: enrolment_prediction_chart.png")
plt.show()

# -- 7. Export Predictions -----------------------------------------------------
results = df.copy()
results["predicted"] = model.predict(scaler.transform(X)).astype(int)
results.to_csv("enrolment_predictions.csv", index=False)
print("Full predictions exported to: enrolment_predictions.csv")
