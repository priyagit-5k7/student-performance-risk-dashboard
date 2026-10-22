
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Student Performance Analytics Dashboard")
st.write(
    "Analyze student data, explore academic patterns, "
    "and compare machine learning models."
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("student-mat.csv", sep=";")
    data = data.drop_duplicates().copy()

    required_columns = [
        "G1", "G2", "G3", "studytime",
        "absences", "sex", "school"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Place student-mat.csv "
        "in the same folder as this Python file."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# Define academic risk for this educational exercise.
# G3 below 10 is classified as at risk.
df["At_Risk"] = (df["G3"] < 10).astype(int)

# --------------------------------------------------
# 3. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

schools = sorted(df["school"].dropna().unique().tolist())

selected_schools = st.sidebar.multiselect(
    "Select school",
    options=schools,
    default=schools
)

if selected_schools:
    filtered_df = df[df["school"].isin(selected_schools)].copy()
else:
    filtered_df = df.iloc[0:0].copy()

if filtered_df.empty:
    st.warning("Select at least one school in the sidebar.")
    st.stop()

# --------------------------------------------------
# 4. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Student Records", len(filtered_df))
col2.metric(
    "Average Final Grade",
    f"{filtered_df['G3'].mean():.2f}/20"
)
col3.metric(
    "At-Risk Records",
    int(filtered_df["At_Risk"].sum())
)
col4.metric(
    "Average Absences",
    f"{filtered_df['absences'].mean():.1f}"
)

st.caption(
    "At risk means a final grade below 10 out of 20 "
    "in this dataset. It is an analytical label, "
    "not a validated prediction of a student's future."
)

# --------------------------------------------------
# 5. DISPLAY DATA
# --------------------------------------------------

st.subheader("Student Dataset")

with st.expander("View student records"):
    st.dataframe(filtered_df, use_container_width=True)

st.subheader("Data Quality")

quality_col1, quality_col2 = st.columns(2)

with quality_col1:
    st.write("Missing values by column")
    missing_data = (
        filtered_df.isnull()
        .sum()
        .rename("Missing Values")
        .to_frame()
    )
    st.dataframe(missing_data, use_container_width=True)

with quality_col2:
    st.write("Dataset summary")
    st.write(f"Rows: {filtered_df.shape[0]}")
    st.write(f"Columns: {filtered_df.shape[1]}")
    st.write(
        "Duplicate rows remaining: "
        f"{filtered_df.duplicated().sum()}"
    )

# --------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.header("Exploratory Data Analysis")

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Final Grade Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        data=filtered_df,
        x="G3",
        bins=15,
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Final Grade (0-20)")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Grades")

    st.pyplot(fig)
    plt.close(fig)

with chart2:
    st.subheader("Study Time and Final Grades")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.boxplot(
        data=filtered_df,
        x="studytime",
        y="G3",
        ax=ax
    )

    ax.set_xlabel("Weekly Study Time Category")
    ax.set_ylabel("Final Grade")
    ax.set_title("Study Time vs Final Grade")

    st.pyplot(fig)
    plt.close(fig)

chart3, chart4 = st.columns(2)

with chart3:
    st.subheader("Academic Risk Distribution")

    risk_counts = (
        filtered_df["At_Risk"]
        .map({0: "Not at risk", 1: "At risk"})
        .value_counts()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    risk_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Students")
    ax.set_title("Academic Risk Categories")
    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)
    plt.close(fig)

with chart4:
    st.subheader("Study Time vs Final Grade")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.scatterplot(
        data=filtered_df,
        x="studytime",
        y="G3",
        hue="At_Risk",
        palette="Set1",
        ax=ax
    )

    ax.set_xlabel("Weekly Study Time Category")
    ax.set_ylabel("Final Grade")
    ax.set_title("Study Time and Risk Classification")

    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# 7. PREPARE DATA FOR MACHINE LEARNING
# --------------------------------------------------

st.header("Machine Learning Model Comparison")

st.write(
    "The target is At_Risk: 1 means final grade below 10; "
    "0 means final grade of 10 or above."
)

# Exclude G1, G2 and G3 to avoid direct grade leakage.
# Also exclude the derived target from input features.
X = filtered_df.drop(
    columns=["G1", "G2", "G3", "At_Risk"]
)

y = filtered_df["At_Risk"]

if y.nunique() < 2:
    st.warning(
        "The selected data contains only one risk category. "
        "Select more schools to train classification models."
    )
    st.stop()

class_counts = y.value_counts()

if class_counts.min() < 2:
    st.warning(
        "There are too few records in one class for a "
        "stratified train/test split. Select more data."
    )
    st.stop()

# Split before fitting preprocessing to avoid leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])

# --------------------------------------------------
# 8. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}

results = []
trained_models = {}
model_predictions = {}

for model_name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    trained_models[model_name] = pipeline
    model_predictions[model_name] = predictions

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test, predictions
        ),
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

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

st.subheader("Performance Metrics")

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# 9. VISUALIZE MODEL COMPARISON
# --------------------------------------------------

st.subheader("Compare Machine Learning Models")

fig, ax = plt.subplots(figsize=(10, 5))

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    ax=ax
)

ax.set_title("Model Performance on the Test Set")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Model")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 10. BEST MODEL AND CONFUSION MATRIX
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_predictions = model_predictions[best_model_name]

st.subheader("Best Model on This Test Split")

st.write(f"Selected by F1 score: **{best_model_name}**")

best_row = results_df.iloc[0]

m1, m2, m3 = st.columns(3)

m1.metric("Accuracy", f"{best_row['Accuracy']:.1%}")
m2.metric("Recall", f"{best_row['Recall']:.1%}")
m3.metric("F1 Score", f"{best_row['F1 Score']:.3f}")

st.write("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not at risk", "At risk"],
    yticklabels=["Not at risk", "At risk"],
    ax=ax
)

ax.set_xlabel("Predicted Category")
ax.set_ylabel("Actual Category")
ax.set_title(f"Confusion Matrix: {best_model_name}")

st.pyplot(fig)
plt.close(fig)

st.write("Classification Report")

report = classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Not at risk", "At risk"],
    output_dict=True,
    zero_division=0
)

st.dataframe(
    pd.DataFrame(report).transpose(),
    use_container_width=True
)

# --------------------------------------------------
# 11. DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("Download Results")

st.download_button(
    label="Download Model Comparison CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison_results.csv",
    mime="text/csv"
)

st.download_button(
    label="Download Filtered Student Data",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="student_analysis_data.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Project: Student Performance Analysis and Academic "
    "Risk Classification | Python, Pandas, Matplotlib, "
    "Seaborn and Scikit-learn"
)

st.warning(
    "Educational prototype using a public Portuguese school "
    "dataset. A single test split on a small dataset does not "
    "establish real-world predictive reliability. This model "
    "should not be used to make consequential decisions about "
    "individual students."
)
