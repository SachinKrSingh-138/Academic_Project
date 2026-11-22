# Student Academic Risk Prediction System

## 1. Project Overview

The **Student Academic Risk Prediction System** is a Machine Learning-based project designed to identify students who may be academically at risk.

The system uses student-related information to predict academic risk and provides an early warning that can help identify students who may need additional academic support.

A Streamlit web application allows users to enter student details, select a Machine Learning algorithm, and view the prediction results.

## 2. Project Objectives

- Identify students who may be academically at risk.
- Apply different Machine Learning classification algorithms.
- Compare the performance of multiple models.
- Estimate academic risk probability.
- Provide basic recommendations based on prediction results.

## 3. Dataset

The project uses the **Student Performance Dataset (Portuguese Language)** from the UCI Machine Learning Repository.

Dataset: `student-por.csv`

- Total records: 649
- Original features: 33
- Input features used: 30
- Target variable: `risk_status`

### Target Variable

The target variable is created using the final grade (`G3`):

- **At Risk:** G3 < 10
- **Not At Risk:** G3 >= 10

The grade columns G1, G2, and G3 are excluded from the input features to avoid target leakage.

Dataset source: https://archive.ics.uci.edu/dataset/320/student+performance

## 4. Machine Learning Algorithms

The project uses three classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Decision Tree model uses sigmoid calibration to improve the quality of predicted probabilities.

Each algorithm has its own saved prediction threshold.

| Algorithm | Threshold |
|---|---:|
| Logistic Regression | 0.45 |
| Decision Tree | 0.15 |
| Random Forest | 0.25 |

## 5. Feature Selection

Feature selection is performed using Mutual Information.

The selected Top 10 features are:

- failures
- school
- higher
- absences
- reason
- Fedu
- Mjob
- Medu
- goout
- freetime

## 6. Model Evaluation

The models are evaluated using Stratified 5-Fold Cross-Validation.

### Cross-Validation Results

| Algorithm | Accuracy | At Risk Precision | At Risk Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 79.77% | 68.78% | 79.36% | 70.81% |
| Random Forest | 82.09% | 69.34% | 74.08% | 70.53% |
| Decision Tree | 75.53% | 37.02% | 83.75% | 67.50% |

### Held-Out Test Set Results

The following results are from the separate test set.

| Algorithm | Accuracy | At Risk Precision | At Risk Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 67.69% | 27.08% | 65.00% | 58.18% |
| Random Forest | 65.38% | 20.93% | 45.00% | 52.86% |
| Calibrated Decision Tree | 64.62% | 25.00% | 65.00% | 55.82% |

The Decision Tree threshold was selected to prioritize At Risk recall. Its saved threshold is 0.15.

## 7. Streamlit Application Features

- Selection between three Machine Learning algorithms.
- Algorithm performance comparison.
- Student information input form.
- Academic risk probability prediction.
- Low, Moderate, and High Risk classification.
- Recommendations based on prediction results.
- Prediction summary.

The application accepts 10 student features directly from the user. The remaining 20 features are filled using dataset median or mode values.

## 8. Project Structure

```text
Academic_Project/
│
├── dataset/
│   └── student-por.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_adjusted.png
│   └── model_comparison.png
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## 10. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd Academic_Project
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 11. Limitations

- The dataset contains records of Portuguese secondary-school students.
- The model's performance may differ on students from other countries or educational institutions.
- Some student features are filled using dataset median or mode values when not entered by the user.
- Predictions are estimates and should not be treated as definitive assessments of a student's academic future.

## 12. Conclusion

This project demonstrates how Machine Learning can be used to develop an early warning system for identifying students who may be academically at risk.

By comparing multiple classification algorithms and providing a simple Streamlit interface, the system demonstrates a practical application of Machine Learning in educational data analysis.