# Heart Failure Prediction Using Machine Learning Techniques

## 📌 Project Overview

This project presents a comprehensive **machine learning–based system for predicting heart failure** using structured clinical data. It was developed as part of the **AIE504 – Machine Learning** course under the **Graduate School at Istanbul Okan University program of Artificial Intelligence Engineering (With Thesis)**.

The work demonstrates an end-to-end machine learning pipeline, including data exploration, preprocessing, feature engineering, model development, evaluation, and optimization, with a strong emphasis on clinical relevance and explainability.

---

## 🎓 Academic Context

* **Course:** AIE504 – Machine Learning
* **Program:** MSc Artificial Intelligence Engineering (With Thesis)
* **Student:** **Chahyaandida Ishaya**
* **Student ID:** 253307014
* **Program Code:** 3307010
* **Instructor:** Assoc. Prof. Dr. Evrim Guler

---

## 🎯 Problem Statement

Heart failure remains a leading cause of global morbidity and mortality. Early detection is critical for improving patient outcomes and reducing healthcare burden. Traditional diagnostic approaches can be time-consuming and subjective.

This project explores how **machine learning models** can assist clinicians by:

* Identifying patients at high risk of heart failure
* Learning hidden patterns in clinical data
* Supporting early intervention and preventive care

---

## 🧠 Objectives

The main objectives of this project are to:

* Build an intelligent heart failure prediction system using supervised learning
* Analyze the influence of key clinical and demographic features
* Compare multiple classification algorithms
* Optimize models using hyperparameter tuning
* Evaluate performance using robust statistical metrics

---

## 📂 Dataset Description

* **Dataset:** Heart Failure / Heart Disease Clinical Records Dataset
* **Source:** Kaggle (publicly available)

### Key Features Include:

* Age, Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol Level
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak (ST Depression)
* ST Segment Slope

**Target Variable:**

* `HeartDisease` (1 = Presence of heart disease, 0 = No heart disease)

---

## 🛠️ Methodology

The project follows a standard machine learning workflow:

1. **Data Acquisition** – Loading and validating the dataset
2. **Exploratory Data Analysis (EDA)** – Statistical summaries, visualizations, correlation analysis
3. **Data Preprocessing**

   * Handling categorical variables using one-hot encoding
   * Feature scaling using StandardScaler
   * Train-test split
4. **Feature Engineering & Selection** – Manual and correlation-based feature selection
5. **Model Development** – Training multiple classifiers
6. **Model Evaluation** – Performance comparison using standard metrics
7. **Hyperparameter Optimization** – GridSearchCV for Random Forest and XGBoost
8. **System Simulation & Visualization** – Confusion matrix, ROC curves, Precision-Recall curves

---

## 🤖 Machine Learning Models Implemented

* Logistic Regression
* Random Forest Classifier (Optimized)
* Support Vector Machine (SVM)
* XGBoost Classifier (Optimized)
* K-Nearest Neighbors (KNN)

---

## 📊 Evaluation Metrics

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Precision–Recall AUC

Visualization tools include:

* Confusion Matrix
* ROC Curves
* Precision–Recall Curves

---

## 🏆 Results Summary

* **Best Performing Models:** Logistic Regression, Optimized XGBoost, and Optimized Random Forest
* **Key Predictive Features:**

  * ST Segment Slope (Flat, Up)
  * Chest Pain Type (ATA, NAP, TA)
  * Exercise-Induced Angina
  * Oldpeak
  * Maximum Heart Rate
  * Fasting Blood Sugar
  * Age

The models achieved strong discriminatory power, demonstrating their effectiveness in identifying patients at risk of heart failure.

---

## 🎥 Literature Review & Paper Presentations

The literature review and conceptual background for this project were presented and recorded as part of the academic requirements. You can access the presentations below:

* **Project Proposal:**
    [https://youtu.be/-2aW7ApoHKo] (https://youtu.be/-2aW7ApoHKo)

* **1st Paper Presentation (Literature Review):**
  [https://youtu.be/NtyqB9kcALE?feature=shared](https://youtu.be/NtyqB9kcALE?feature=shared)

* **2nd Paper Presentation (Extended Review & Discussion):**
  [https://youtu.be/GvjxzxGNIPI?feature=shared](https://youtu.be/GvjxzxGNIPI?feature=shared)

---

## 📁 Repository Structure

```
├── heart_failure_prediction_ml_AIE504_project.ipynb
├── heart_failure_prediction_ml_aie504_project.py
├── README.md
```

---

## 🚀 How to Run the Project

1. Clone the repository
2. Install required dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
   ```
3. Open and run the Jupyter Notebook or Python script
4. Review outputs, visualizations, and evaluation results

---

## 📌 Conclusion

This project demonstrates the practical application of machine learning in healthcare analytics. By combining robust data preprocessing, multiple predictive models, and rigorous evaluation, the system shows strong potential as a **clinical decision support tool** for early heart failure detection.

The work highlights how explainable, data-driven approaches can enhance diagnostic accuracy and contribute meaningfully to preventive healthcare.

---

## 👤 Author

**Chahyaandida Ishaya**
MSc Artificial Intelligence Engineering (With Thesis), Istanbul Okan University
Machine Learning | Data Science | Health Analytics
