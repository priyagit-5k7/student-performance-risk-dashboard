# student-performance-risk-dashboard
Interactive student performance analytics dashboard using Python, Streamlit, and machine learning to analyze academic risk and compare classification models.

# Student Performance & Academic Risk Prediction Dashboard

## 📌 Project Overview

The **Student Performance & Academic Risk Prediction Dashboard** is a data science and machine learning project developed to analyze student academic performance, explore factors associated with academic outcomes, and identify students who may be at risk of academic underperformance.

Built using Python, Pandas, Scikit-learn, Matplotlib, Seaborn, and Streamlit, the project combines data preprocessing, exploratory data analysis (EDA), data quality assessment, machine learning model comparison, and interactive data visualization into a unified dashboard.

The system uses student performance data to classify students as **potentially at risk** or **not at risk**, based on their final mathematics grade. It compares Logistic Regression, Decision Tree, and Random Forest classifiers to evaluate their predictive performance and provide insights into the strengths and limitations of different machine learning approaches.

The project aims to demonstrate how data analytics and machine learning can support evidence-based academic monitoring and help educators investigate students who may benefit from additional academic support.

## 🎯 Project Objectives

- Analyze student demographic, family, lifestyle, and educational attributes.
- Perform exploratory data analysis to understand the dataset and academic performance patterns.
- Assess data quality by checking missing values, duplicate records, grade validity, and potential statistical outliers.
- Visualize student performance distributions and patterns using interactive charts.
- Develop a binary classification model to identify students potentially at risk of academic underperformance.
- Compare multiple machine learning algorithms using Accuracy, Precision, Recall, and F1 Score.
- Display model evaluation results and classification insights in a user-friendly dashboard.
- Demonstrate a complete data science workflow, from raw dataset exploration to model evaluation and application deployment.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Interactive Dashboard:** Streamlit
- **Dataset Format:** CSV
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud (planned or deployed, depending on current status)

## 📊 Dataset Description

The project uses the Student Performance dataset containing mathematics-course records from secondary school students.

- **Records:** 395 students
- **Original attributes:** 33 columns
- **Target variable:** `At_Risk`
- **Risk definition:** Students with a final mathematics grade (`G3`) below 10 out of 20 are classified as at risk; students with a grade of 10 or above are classified as not at risk.

The dataset contains attributes related to student demographics, family background, parental education, study habits, school-related factors, lifestyle, absences, and academic grades.

The target variable is derived from the final grade. To reduce target leakage, the model excludes `G1`, `G2`, and `G3` from its predictor features.

## ✨ Key Features

### 1. Data Quality Analysis
- Checks dataset dimensions and column information.
- Identifies missing values and duplicate records.
- Validates grade values against the expected 0–20 range.
- Detects potential outliers using the Interquartile Range (IQR) method.
- Generates statistical summaries and data quality reports.

### 2. Exploratory Data Analysis
- Examines the distribution of student academic risk.
- Analyzes numerical and categorical attributes.
- Visualizes academic performance patterns.
- Explores relationships between study habits, absences, and student grades.

### 3. Academic Risk Classification
- Creates a binary academic-risk target based on the final grade.
- Separates predictor features from the target variable.
- Splits the data into training and testing sets.
- Uses preprocessing pipelines for numerical and categorical features.
- Applies machine learning classifiers to predict potential academic risk.

### 4. Machine Learning Model Comparison

Three classification algorithms are evaluated:

- **Logistic Regression:** Establishes a baseline for binary classification.
- **Decision Tree:** Models potentially nonlinear relationships through decision rules.
- **Random Forest:** Combines multiple decision trees to improve predictive performance.

The models are compared using Accuracy, Precision, Recall, and F1 Score. The highest F1 Score is used to select the best-performing model among the evaluated candidates.

### 5. Interactive Dashboard
- Presents student performance information through visual summaries.
- Displays academic risk distributions and performance patterns.
- Supports exploration of student data using available dashboard filters.
- Presents machine learning evaluation results and classification insights.
- Helps users interpret patterns in student academic performance.

### 6. Visual Reports and Exports
The analysis generates the following outputs:

- `risk_distribution.png` — distribution of students by academic-risk category.
- `model_comparison.png` — comparison of classification model metrics.
- `confusion_matrix.png` — visualization of correct and incorrect predictions.
- `model_comparison_results.csv` — tabular model evaluation results.
- `data_quality_outputs/` — data quality reports and visualizations generated by the data quality analysis script.

## 📈 Initial Model Evaluation Results

The current evaluation uses a stratified train-test split with a 25% test set, corresponding to 99 students.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 72.7% | 62.5% | 45.5% | 52.6% |
| Logistic Regression | 67.7% | 52.6% | 30.3% | 38.5% |
| Decision Tree | 68.7% | 58.3% | 21.2% | 31.1% |

**Best-performing model in the current test run: Random Forest**, with an F1 Score of 52.6%.

These are initial results from one train-test split, not a guarantee of future performance. In particular, the Random Forest model's recall of 45.5% indicates that it misses some students classified as at risk. Further evaluation, cross-validation, threshold tuning, and recall-focused improvements are recommended before using the model for practical academic interventions.

## ⚙️ Project Workflow

1. Load the student performance dataset.
2. Inspect dataset structure and data quality.
3. Clean and validate relevant data.
4. Conduct exploratory data analysis.
5. Define the academic-risk classification target.
6. Separate features and target variable while excluding grade leakage.
7. Split the dataset into training and testing sets.
8. Preprocess numerical and categorical features.
9. Train Logistic Regression, Decision Tree, and Random Forest models.
10. Evaluate and compare model performance.
11. Generate charts, reports, and evaluation outputs.
12. Explore the results through the Streamlit dashboard.

## 📂 Project Structure

```text
student-performance-risk-dashboard/
├── student-mat.csv
├── real_student_dashboard.py
├── model_comparison.py
├── data_quality_analysis.py
├── requirements.txt
├── risk_distribution.png
├── student_performance.png
├── model_comparison.png
├── confusion_matrix.png
├── model_comparison_results.csv
├── data_quality_outputs/
└── README.md
```

*The file list may vary depending on which generated reports and visualizations are included in the repository.*

## 🚀 Installation and Usage

### Prerequisites
- Python 3.10 or a compatible Python version
- pip package manager
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-risk-dashboard.git
cd student-performance-risk-dashboard
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Data Quality Analysis

```bash
python data_quality_analysis.py
```

### 4. Run the Model Comparison

```bash
python model_comparison.py
```

This trains and evaluates the classification models and generates the model comparison table and associated visual outputs.

### 5. Launch the Dashboard

```bash
python -m streamlit run real_student_dashboard.py
```

Open the local URL displayed in the terminal to interact with the dashboard.

## 🔮 Future Enhancements

- Improve recall for identifying potentially at-risk students.
- Apply stratified cross-validation for more reliable model evaluation.
- Tune classification thresholds based on the cost of missing at-risk students.
- Add feature-importance and model-interpretability visualizations.
- Incorporate additional academic data to evaluate model generalization.
- Enhance dashboard filters, reporting, and export capabilities.
- Add automated tests and deployment checks.
- Deploy the dashboard publicly using Streamlit Community Cloud.

## ⚠️ Limitations and Responsible Use

This project is an educational data science demonstration. The dataset is relatively small, and the model's performance has been evaluated using a single train-test split. Results may not generalize to different schools, regions, or student populations.

Predictions should not be treated as definitive judgments about individual students. Academic risk predictions should be used only as indicators for further review and supportive interventions, with appropriate human oversight and attention to privacy, fairness, and data quality.

## 🎓 Learning Outcomes

This project demonstrates practical experience in:

- Python-based data analysis and preprocessing.
- Exploratory data analysis and visualization.
- Data quality assessment and statistical outlier detection.
- Machine learning classification and evaluation.
- Feature preprocessing using Scikit-learn pipelines.
- Comparing classification algorithms using multiple metrics.
- Developing interactive analytical dashboards using Streamlit.
- Organizing and documenting a reproducible data science project.

## 👩‍💻 Project Information

**Project Title:** Student Performance & Academic Risk Prediction Dashboard  
**Domain:** Data Science and Machine Learning  
**Primary Language:** Python  
**Application Framework:** Streamlit  
**Purpose:** Academic performance analysis and early identification of potential academic risk.

---

*Disclaimer: This project is intended for educational and analytical purposes. It does not replace professional educational assessment or individualized student support.*
