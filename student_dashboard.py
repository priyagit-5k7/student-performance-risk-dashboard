
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Performance Analysis")
st.write("Analyze academic performance and predict student outcomes.")

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 6, 3, 9],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 52, 85, 62, 95],
    "Marks": [30, 35, 40, 45, 55, 65, 72, 90, 32, 78, 42, 95]
}

df = pd.DataFrame(data)
df = df.drop_duplicates()

# Key performance indicators
col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))
col2.metric("Average Marks", f"{df['Marks'].mean():.1f}")
col3.metric("Average Attendance", f"{df['Attendance'].mean():.1f}%")

st.subheader("Student Dataset")
st.dataframe(df, use_container_width=True)

# Visualization
st.subheader("Study Hours vs Marks")

fig, ax = plt.subplots()
ax.scatter(df["Study_Hours"], df["Marks"])
ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks")
ax.set_title("Relationship Between Study Hours and Marks")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# Train classification model
X = df[["Study_Hours", "Attendance"]]
y = (df["Marks"] >= 50).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

evaluation_model = LogisticRegression()
evaluation_model.fit(X_train, y_train)

predictions = evaluation_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Evaluation")
st.metric("Test Accuracy", f"{accuracy:.1%}")
st.caption(
    "This score uses a small illustrative dataset and is not "
    "evidence of real-world predictive performance."
)

# Train final model on all available sample data
model = LogisticRegression()
model.fit(X, y)

# Interactive prediction
st.subheader("Predict Student Performance")

study_hours = st.slider(
    "Daily study hours", 0.0, 12.0, 5.0, 0.5
)

attendance = st.slider(
    "Attendance (%)", 0, 100, 75
)

if st.button("Predict Performance"):
    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance]
    })

    result = model.predict(new_student)[0]
    probability = model.predict_proba(new_student)[0][1]

    if result == 1:
        st.success("Predicted category: Good")
    else:
        st.warning("Predicted category: Needs Improvement")

    st.write(
        f"Model-estimated probability of the 'Good' category: "
        f"{probability:.1%}"
    )

    st.caption(
        "This is a demonstration using synthetic data, not a "
        "reliable assessment of an actual student."
    )

st.subheader("Key Observation")
st.write(
    "Explore how study hours and attendance relate to marks "
    "in this sample. A larger, representative dataset is needed "
    "to draw reliable conclusions."
)

st.info("Project developed using Python, Pandas, Matplotlib, "
        "Scikit-learn, and Streamlit.")

