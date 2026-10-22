
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load real data
df = pd.read_csv("student-mat.csv", sep=";")

print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum().sum())

# 2. Prepare the target
# G3 below 10 = at risk; G3 of 10 or more = not at risk
df["At_Risk"] = (df["G3"] < 10).astype(int)

# Exclude G1, G2 and G3 to avoid using grades
# that could reveal the final outcome.
X = df.drop(columns=["G1", "G2", "G3", "At_Risk"])
y = df["At_Risk"]

# 3. Explore the data
print("\nStudent risk distribution:")
print(y.map({0: "Not at risk", 1: "At risk"}).value_counts())

sns.countplot(x=y)
plt.title("Student Academic Risk Distribution")
plt.xlabel("At Risk (1 = Yes, 0 = No)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("risk_distribution.png")
plt.show()

# 4. Split data before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Identify numerical and categorical columns
numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 6. Compare three classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}

results = []
trained_models = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

    trained_models[name] = pipeline

# 7. Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("model_comparison_results.csv", index=False)

# 8. Plot model performance
plot_df = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

plot_df.plot(kind="bar", figsize=(10, 5))
plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# 9. Evaluate the best model by F1 score
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = best_model.predict(X_test)

print("\nBest model by test F1 score:", best_name)
print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    labels=[0, 1],
    display_labels=["Not at risk", "At risk"]
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nGenerated files:")
print("- risk_distribution.png")
print("- model_comparison.png")
print("- confusion_matrix.png")
print("- model_comparison_results.csv")
