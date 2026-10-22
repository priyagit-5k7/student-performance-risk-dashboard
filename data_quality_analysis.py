
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path(__file__).parent / "student-mat.csv"
OUTPUT_DIR = Path(__file__).parent / "data_quality_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load the dataset
df = pd.read_csv(DATA_FILE, sep=";")

print("=" * 60)
print("STUDENT PERFORMANCE - DATA QUALITY ANALYSIS")
print("=" * 60)

# 2. Dataset overview
print("\n1. DATASET OVERVIEW")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# 3. Missing-value analysis
print("\n2. MISSING VALUES")
missing = df.isnull().sum().sort_values(ascending=False)
missing_report = pd.DataFrame({
    "Column": missing.index,
    "Missing_Values": missing.values,
    "Missing_Percentage": (
        missing.values / len(df) * 100
    ).round(2)
})
print(missing_report.to_string(index=False))
missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv", index=False
)

# 4. Check grade validity
print("\n3. GRADE VALIDATION")
for column in ["G1", "G2", "G3"]:
    invalid = df[~df[column].between(0, 20)]
    print(f"{column}: {len(invalid)} invalid values")

# 5. Check numerical outliers using the IQR method
print("\n4. NUMERICAL OUTLIER ANALYSIS")
numeric_columns = ["age", "absences", "G1", "G2", "G3"]
outlier_results = []

for column in numeric_columns:
    values = df[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_bound) | (values > upper_bound)
    ]

    outlier_results.append({
        "Column": column,
        "Q1": q1,
        "Q3": q3,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": len(outliers)
    })

    print(
        f"{column}: {len(outliers)} potential outliers "
        f"(bounds: {lower_bound:.2f} to {upper_bound:.2f})"
    )

outlier_report = pd.DataFrame(outlier_results)
outlier_report.to_csv(
    OUTPUT_DIR / "outlier_report.csv", index=False
)

# 6. Numerical summary
print("\n5. NUMERICAL SUMMARY")
print(df[numeric_columns].describe().round(2).to_string())

df[numeric_columns].describe().round(2).to_csv(
    OUTPUT_DIR / "numerical_summary.csv"
)

# 7. Create an outlier-count chart
plt.figure(figsize=(9, 5))
plt.bar(
    outlier_report["Column"],
    outlier_report["Outlier_Count"]
)
plt.title("Potential Outliers by Numerical Column")
plt.xlabel("Column")
plt.ylabel("Number of Potential Outliers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "outlier_counts.png", dpi=150)
plt.close()

# 8. Create a boxplot for numerical columns
plt.figure(figsize=(9, 5))
df[numeric_columns].boxplot()
plt.title("Boxplot of Student Numerical Data")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "numerical_boxplots.png", dpi=150)
plt.close()

print("\n6. OUTPUT FILES")
print(f"Reports and charts saved to: {OUTPUT_DIR.resolve()}")
print("\nAnalysis completed successfully.")
