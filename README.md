# 🛳️ Titanic Survival Intelligence

An **interactive machine learning dashboard** built with **Streamlit and Plotly** that analyzes and predicts passenger survival probabilities from the Titanic dataset.  
The application provides **data-driven insights, visual analytics, and ML-powered predictions** with a modern dark-themed UI.

---

## 📌 Project Overview

**Titanic Survival Intelligence** is designed to:
- Explore passenger demographics and survival trends
- Predict survival probability using a trained ML model
- Visualize insights interactively using filters
- Present results through a clean, professional dashboard

This project combines **data preprocessing, machine learning, and interactive visualization** into a single end-to-end application.

---

## ✨ Key Features

### 🔍 Interactive Filters
- Passenger Class (1st, 2nd, 3rd)
- Gender
- Age range slider

### 📊 Data Visualization
- Age Distribution (Histogram)
- Passenger Class Distribution (Donut Chart)
- Gender Distribution (Bar Chart)
- Actual Survival Rate (Gauge Chart)

### 🔮 ML-Powered Predictions
- Survival probability for each passenger
- Categorized as:
  - 🟢 High Survival (≥70%)
  - 🟡 Medium Survival (40–70%)
  - 🔴 Low Survival (<40%)

### 💡 Insights & Analysis
- Survival probability by gender and class
- Age vs survival probability scatter plot
- Key findings with survival rates:
  - Female vs Male survival
  - Survival by passenger class

### 🌙 Dark Theme UI
- Mobile-style pure black theme
- Styled metric cards, charts, and tabs
- Plotly graphs fully adapted to dark mode

---

## 🧠 Machine Learning Model

- **Algorithm:** Trained classification model (Scikit-learn)
- **Input Features:** Passenger class, gender, age, family size, etc.
- **Output:** Survival probability (0–1)
- **Model File:** `models/titanic_model.pkl`

---

## 🗂 Project Structure

