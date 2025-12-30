import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


st.title("🎓 Student Performance Prediction")
st.write("Predict student academic performance using machine learning.")

# Load model
model = joblib.load("model/student_performance_model.pkl")


@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/student-mat.csv", sep=";")

    def performance_label(score):
        if score <= 10:
            return "Low"
        elif score <= 15:
            return "Medium"
        else:
            return "High"

    df["performance"] = df["G3"].apply(performance_label)

    X_ref = df.drop(["G3", "performance"], axis=1)
    y_ref = df["performance"]

    return X_ref, y_ref

X_ref, y_ref = load_reference_data()


st.subheader("Enter Student Details")

gender = st.selectbox("Gender", ["M", "F"])
studytime = st.slider("Study Time (1 = low, 4 = high)", 1, 4, 2)
absences = st.slider("Number of Absences", 0, 50, 5)
internet = st.selectbox("Internet Access", ["yes", "no"])
parent_edu = st.selectbox(
    "Mother's Education (0 = none, 4 = higher)",
    [0, 1, 2, 3, 4]
)

g1 = st.slider("Previous Period Score (G1)", 0, 20, 12)
g2 = st.slider("Second Period Score (G2)", 0, 20, 13)

# Take input data
input_data = {
    'school': 'GP',
    'sex': gender,
    'age': 17,
    'address': 'U',
    'famsize': 'GT3',
    'Pstatus': 'T',
    'Medu': parent_edu,
    'Fedu': 2,
    'Mjob': 'other',
    'Fjob': 'other',
    'reason': 'course',
    'guardian': 'mother',
    'traveltime': 1,
    'studytime': studytime,
    'failures': 0,
    'schoolsup': 'no',
    'famsup': 'yes',
    'paid': 'no',
    'activities': 'yes',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': internet,
    'romantic': 'no',
    'famrel': 4,
    'freetime': 3,
    'goout': 3,
    'Dalc': 1,
    'Walc': 1,
    'health': 4,
    'absences': absences,
    'G1': g1,
    'G2': g2
}

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict Performance"):
    prediction = model.predict(input_df)[0]

    st.subheader("📌 Prediction Result")

    if prediction == "Low":
        st.error("🔴 Low Performance")
        st.write("Suggestions:")
        st.write("- Increase daily study time")
        st.write("- Reduce absences")
        st.write("- Improve consistency")

    elif prediction == "Medium":
        st.warning("🟡 Medium Performance")
        st.write("Suggestions:")
        st.write("- Slightly increase study hours")
        st.write("- Maintain attendance")
        st.write("- Practice weak subjects")

    else:
        st.success("🟢 High Performance")
        st.write("Great job! Keep up the good work 👏")


def compute_feature_importance(model, X, y):
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": result.importances_mean
    }).sort_values(by="Importance", ascending=False)

    return importance_df

st.divider()
st.subheader("📊 Feature Importance (Global Model Explanation)")

importance_df = compute_feature_importance(model, X_ref, y_ref)

fig, ax = plt.subplots()
ax.barh(
    importance_df["Feature"].head(10),
    importance_df["Importance"].head(10)
)
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title("Top 10 Most Important Features")

st.pyplot(fig)
