
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create a sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 6, 3, 9],
    "Attendance":  [50, 55, 60, 65, 70, 75, 80, 90, 52, 85, 62, 95],
    "Marks":       [30, 35, 40, 45, 55, 65, 72, 90, 32, 78, 42, 95]
}

df = pd.DataFrame(data)

# 2. Inspect and clean the data
print("Dataset:")
print(df)
print("\nMissing values:")
print(df.isnull().sum())

df = df.drop_duplicates()

# 3. Analyze the data
print("\nAverage marks:", df["Marks"].mean())
print("Highest marks:", df["Marks"].max())
print("Average attendance:", df["Attendance"].mean())

# 4. Visualize study hours vs marks
plt.scatter(df["Study_Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Student Marks")
plt.grid(True)
plt.tight_layout()
plt.savefig("student_performance.png")
plt.show()

# 5. Build a simple classification model
# Label: 1 = Good (marks >= 50), 0 = Needs Improvement
X = df[["Study_Hours", "Attendance"]]
y = (df["Marks"] >= 50).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nModel accuracy:", accuracy_score(y_test, predictions))

# 6. Predict for a new student
new_student = pd.DataFrame({
    "Study_Hours": [5],
    "Attendance": [75]
})

result = model.predict(new_student)[0]

print(
    "Prediction:",
    "Good" if result == 1 else "Needs Improvement"
)
