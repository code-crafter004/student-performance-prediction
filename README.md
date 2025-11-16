
Student Performance Prediction using PySpark MLlib
-----------------------------------------------------------

A machine learning system built using PySpark MLlib to predict student academic performance based on:

Attendance

Marks

Engagement Score

The goal is to identify at-risk students early, enabling timely academic intervention.

Project Overview
------------------

This project builds a full machine learning pipeline in PySpark, including:

Data preprocessing

Feature engineering

Logistic Regression model training

Model evaluation

Model saving/loading

Streamlit-based web dashboard

Tech Stack
------------

PySpark 3.5.1

Python 3.x

Streamlit (for UI)

Pandas / NumPy

MLlib (Logistic Regression)

GitHub Codespaces (dev environment)



student-performance-prediction/
│
├── data/
│   └── student_performance.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_inference.py
│   ├── utils.py
│   └── __init__.py
│
├── app/
│   ├── streamlit_app.py
│   └── __init__.py
│
├── models/
│   └── logistic_regression_model/      # Auto-generated after training
│
├── main.py
├── requirements.txt
├── README.md
└── notebooks/
    └── exploratory_analysis.ipynb


📊 Dataset

The dataset contains:

Column	                    Description
attendance	                Attendance percentage
marks	                    Total exam score
engagement_score	        Participation / activity score
final_result	            Pass/Fail (target variable)

