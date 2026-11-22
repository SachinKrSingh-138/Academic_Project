
# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.feature_selection import mutual_info_classif


# --------------------------------------------------
# 1. Define the Mutual Information Function
# --------------------------------------------------

# These are the numerical features used during preprocessing
numerical_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences"
]


# Define the Mutual Information function required by saved models
def mutual_information_scores(X, y):

    # Identify numerical and categorical encoded features
    discrete_mask = (
        [False] * len(numerical_cols)
        + [True] * (X.shape[1] - len(numerical_cols))
    )

    return mutual_info_classif(
        X,
        y,
        discrete_features=discrete_mask,
        random_state=42
    )


# --------------------------------------------------
# 2. Configure the Streamlit Application
# --------------------------------------------------

# Set the page configuration
st.set_page_config(
    page_title="Student Academic Risk Prediction System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Academic Risk Prediction System")

st.write(
    "An Early Warning System that uses Machine Learning "
    "to identify students who may be academically at risk."
)


# --------------------------------------------------
# 3. Load the Dataset and Trained Model Bundle
# --------------------------------------------------

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent

# Load the student dataset
dataset_path = BASE_DIR / "dataset" / "student-por.csv"

# Load the trained model bundle
model_path = BASE_DIR / "models" / "best_model.pkl"


@st.cache_data
def load_dataset():

    # Read the student performance dataset
    return pd.read_csv(dataset_path, sep=";")


@st.cache_resource
def load_model():

    # Load all trained models, thresholds and evaluation metrics
    return joblib.load(model_path)


try:
    dataset = load_dataset()
    model_bundle = load_model()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# Extract model information from the bundle
models = model_bundle["models"]
thresholds = model_bundle["thresholds"]
metrics = model_bundle["metrics"]

at_risk_class = model_bundle["at_risk_class"]
not_at_risk_class = model_bundle["not_at_risk_class"]

class_labels = model_bundle["class_labels"]


# --------------------------------------------------
# 4. Prepare the Dataset for Student Input
# --------------------------------------------------

# Remove grade columns and target information from input features
feature_columns = [
    column for column in dataset.columns
    if column not in ["G1", "G2", "G3", "risk_status"]
]

# Keep the original feature order
reference_data = dataset[feature_columns].copy()

# Identify numerical and categorical features
numeric_features = reference_data.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = [
    column for column in feature_columns
    if column not in numeric_features
]

# Calculate default values for features not entered by the user
default_values = {}

for column in feature_columns:

    # Use median for numerical features
    if column in numeric_features:
        default_values[column] = reference_data[column].median()

    # Use mode for categorical features
    else:
        default_values[column] = reference_data[column].mode()[0]


# --------------------------------------------------
# 5. Display Model Performance Comparison
# --------------------------------------------------

st.header("📊 Model Performance Comparison")

st.write(
    "Compare the evaluation performance of the available "
    "Machine Learning algorithms."
)

# Prepare the performance comparison table
performance_data = []

for model_name in models.keys():

    # Get the saved evaluation metrics
    model_metrics = metrics[model_name]

    performance_data.append({
        "Algorithm": model_name,
        "Accuracy": model_metrics["accuracy"] * 100,
        "At Risk Precision": model_metrics["precision"] * 100,
        "At Risk Recall": model_metrics["recall"] * 100,
        "Macro F1": model_metrics["macro_f1"] * 100
    })

performance_df = pd.DataFrame(performance_data)

# Display the accuracy comparison graph
chart_data = performance_df.set_index("Algorithm")[["Accuracy"]]

st.subheader("Algorithm Accuracy Comparison")

st.bar_chart(chart_data)

# Display detailed evaluation metrics
st.subheader("Detailed Model Performance")

display_df = performance_df.copy()

display_df["Accuracy"] = display_df["Accuracy"].map(
    lambda value: f"{value:.2f}%"
)

display_df["At Risk Precision"] = display_df["At Risk Precision"].map(
    lambda value: f"{value:.2f}%"
)

display_df["At Risk Recall"] = display_df["At Risk Recall"].map(
    lambda value: f"{value:.2f}%"
)

display_df["Macro F1"] = display_df["Macro F1"].map(
    lambda value: f"{value:.2f}%"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Evaluation metrics are based on the values saved in the model bundle. "
    "They should represent the same evaluation method for each model."
)


# --------------------------------------------------
# 6. Student Information and Algorithm Selection
# --------------------------------------------------

st.header("🧑‍🎓 Student Academic Risk Assessment")

st.write(
    "Enter the student's details and select an algorithm "
    "to calculate the academic risk probability."
)

with st.form("student_prediction_form"):

    # Select the Machine Learning algorithm
    selected_model_name = st.selectbox(
        "Select Machine Learning Algorithm",
        list(models.keys())
    )

    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:

        # Select school
        school = st.selectbox(
            "School",
            sorted(reference_data["school"].dropna().unique())
        )

        # Select whether the student wants higher education
        higher = st.selectbox(
            "Interested in Higher Education",
            sorted(reference_data["higher"].dropna().unique())
        )

        # Enter previous class failures
        failures = st.number_input(
            "Number of Previous Failures",
            min_value=0,
            max_value=4,
            value=0,
            step=1
        )

        # Enter the number of absences
        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=int(default_values["absences"]),
            step=1
        )

        # Select the reason for choosing the school
        reason = st.selectbox(
            "Reason for Choosing School",
            sorted(reference_data["reason"].dropna().unique())
        )

    with col2:

        # Select father's education level
        Fedu = st.selectbox(
            "Father's Education Level",
            [0, 1, 2, 3, 4],
            index=2
        )

        # Select mother's occupation
        Mjob = st.selectbox(
            "Mother's Occupation",
            sorted(reference_data["Mjob"].dropna().unique())
        )

        # Select mother's education level
        Medu = st.selectbox(
            "Mother's Education Level",
            [0, 1, 2, 3, 4],
            index=2
        )

        # Select student's social activity level
        goout = st.selectbox(
            "Going Out with Friends (1-5)",
            [1, 2, 3, 4, 5],
            index=2
        )

        # Select student's free time level
        freetime = st.selectbox(
            "Free Time After School (1-5)",
            [1, 2, 3, 4, 5],
            index=2
        )

    # Submit the student information
    submit_button = st.form_submit_button(
        "Predict Academic Risk",
        use_container_width=True
    )


# --------------------------------------------------
# 7. Generate Student Risk Prediction
# --------------------------------------------------

if submit_button:

    # Start with default values for all 30 input features
    input_values = default_values.copy()

    # Replace defaults with the student's entered information
    input_values.update({
        "school": school,
        "higher": higher,
        "failures": failures,
        "absences": absences,
        "reason": reason,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Medu": Medu,
        "goout": goout,
        "freetime": freetime
    })

    # Create input DataFrame in the original feature order
    input_data = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    # Get the selected trained model
    selected_model = models[selected_model_name]

    # Get the threshold saved for the selected algorithm
    selected_threshold = thresholds[selected_model_name]

    # Calculate prediction probabilities
    probabilities = selected_model.predict_proba(input_data)

    # Identify the correct probability column for At Risk
    risk_column = list(selected_model.classes_).index(
        at_risk_class
    )

    # Extract the At Risk probability
    at_risk_probability = probabilities[0][risk_column]

    # Apply the selected model's threshold
    if at_risk_probability >= selected_threshold:

        prediction = at_risk_class

    else:

        prediction = not_at_risk_class

    # Convert probability into percentage
    risk_percentage = at_risk_probability * 100

    # Convert the predicted class into readable text
    if prediction == at_risk_class:

        prediction_text = "At Risk"

    else:

        prediction_text = "Not At Risk"


    # --------------------------------------------------
    # 8. Display Prediction Results
    # --------------------------------------------------

    st.divider()

    st.header("📋 Prediction Result")

    st.write(f"**Algorithm used:** {selected_model_name}")

    st.write(
        f"**Selected Threshold:** {selected_threshold:.2f}"
    )

    st.metric(
        "Academic Risk Probability",
        f"{risk_percentage:.2f}%"
    )

    if prediction_text == "At Risk":

        st.error(
            "Student predicted as At Risk."
        )

    else:

        st.success(
            "Student predicted as Not At Risk."
        )


    # --------------------------------------------------
    # 9. Display Student Risk Level
    # --------------------------------------------------

    st.subheader("Student Risk Level")

    if risk_percentage < 40:

        risk_level = "Low Risk"

        st.success(
            f"{risk_level} — {risk_percentage:.2f}%"
        )

    elif risk_percentage < 70:

        risk_level = "Moderate Risk"

        st.warning(
            f"{risk_level} — {risk_percentage:.2f}%"
        )

    else:

        risk_level = "High Risk"

        st.error(
            f"{risk_level} — {risk_percentage:.2f}%"
        )


    # --------------------------------------------------
    # 10. Display Recommendations
    # --------------------------------------------------

    st.subheader("💡 Recommendations")

    if prediction_text == "At Risk":

        st.write(
            "- Provide additional academic support and guidance."
        )

        st.write(
            "- Monitor the student's attendance regularly."
        )

        st.write(
            "- Identify subjects or topics where the student needs help."
        )

        st.write(
            "- Encourage regular study habits and communication with teachers."
        )

    else:

        st.write(
            "- Continue maintaining regular study habits."
        )

        st.write(
            "- Maintain consistent attendance."
        )

        st.write(
            "- Continue monitoring academic progress."
        )


    # --------------------------------------------------
    # 11. Display Prediction Summary
    # --------------------------------------------------

    st.subheader("Prediction Summary")

    summary_data = {
        "Parameter": [
            "Selected Algorithm",
            "Threshold",
            "Academic Risk Probability",
            "Predicted Status",
            "Risk Level"
        ],

        "Result": [
            selected_model_name,
            f"{selected_threshold:.2f}",
            f"{risk_percentage:.2f}%",
            prediction_text,
            risk_level
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 12. Important Information
# --------------------------------------------------

st.divider()

st.caption(
    "Note: This system is an educational Machine Learning project. "
    "The dataset contains Portuguese secondary-school student records. "
    "The remaining features not entered by the user are filled using "
    "dataset median or mode values. Predictions should not be treated "
    "as a definitive assessment of a student's academic future."
)
