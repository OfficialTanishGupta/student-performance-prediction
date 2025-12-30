🎓 Student Performance Prediction

🔗 Live Application:
👉 https://student-performance-predic.streamlit.app

📌 Overview

Student Performance Prediction is an end-to-end machine learning project that predicts a student’s academic performance (Low / Medium / High) based on academic history, study habits, and socio-demographic factors.

The project demonstrates the complete machine learning lifecycle — from data analysis and model training to deployment and explainability — through an interactive web application.

🎯 Problem Statement

Educational institutions often lack early-warning systems to identify students who may need additional academic support.

This project aims to:

Predict student performance using machine learning

Identify key factors influencing academic outcomes

Provide an interactive and explainable AI-based solution

🚀 Key Features

End-to-end ML pipeline

Data preprocessing using ColumnTransformer

Model training and evaluation with multiple classifiers

Interactive web application using Streamlit

Real-time performance prediction

Explainable AI using permutation feature importance

Live deployment on Streamlit Community Cloud

🧠 Machine Learning Workflow
📊 Dataset

Student Performance Dataset (UCI Machine Learning Repository / Kaggle)

Contains academic, demographic, and behavioral attributes

🏷 Target Variable

Student performance is categorized into:

Final Score	Performance Label
0 – 10	Low
11 – 15	Medium
16 – 20	High
⚙️ Data Preprocessing

Numerical features scaled using StandardScaler

Categorical features encoded using OneHotEncoder

Unified preprocessing using ColumnTransformer

Prevents data leakage and ensures consistency

🤖 Model Training

Multiple models were trained and evaluated:

Logistic Regression

Random Forest Classifier ✅ (selected)

Gradient Boosting Classifier

The Random Forest model showed the best performance and was selected for deployment.

📈 Model Explainability

Implemented Permutation Feature Importance

Provides global explanation of model behavior

Key influential features include:

Previous grades (G1, G2)

Study time

Absences

Academic failures

🖥️ Web Application

The Streamlit web app allows users to:

Enter student details

Predict academic performance

Receive actionable improvement suggestions

Visualize feature importance for model transparency

🔗 Live App:
👉 https://student-performance-predic.streamlit.app

📂 Project Structure
student-performance-prediction/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── data/
│   └── student-mat.csv     # Dataset
├── model/
│   └── student_performance_model.pkl
└── notebooks/
    └── eda.ipynb           # Exploratory Data Analysis

🛠️ Tech Stack

Programming Language: Python

Libraries: pandas, numpy, scikit-learn, matplotlib, joblib

Web Framework: Streamlit

Deployment: Streamlit Community Cloud

Version Control: Git & GitHub

▶️ How to Run Locally

Clone the repository

git clone https://github.com/OfficialTanishGupta/student-performance-prediction.git
cd student-performance-prediction


Install dependencies

pip install -r requirements.txt


Run the application

python -m streamlit run app.py

💡 Key Learnings

Built an industry-standard ML pipeline using Pipeline and ColumnTransformer

Learned to prevent data leakage in preprocessing

Understood feature dominance and model bias

Implemented explainable AI for model transparency

Gained hands-on experience in deploying ML models

🎤 Interview Talking Points

“I developed and deployed an end-to-end machine learning application that predicts student performance and explains predictions using permutation feature importance. The application is live and publicly accessible.”

🔮 Future Enhancements

Add prediction confidence/probability scores

Train a model without previous grades for early-stage prediction

Implement per-student (local) explanations

Improve UI with advanced visual analytics

Integrate database support for tracking predictions

🙌 Acknowledgements

UCI Machine Learning Repository

Streamlit Community Cloud

scikit-learn Documentation

📬 Contact

Tanish Gupta
CSE Graduate | AI/ML Enthusiast
GitHub: https://github.com/OfficialTanishGupta