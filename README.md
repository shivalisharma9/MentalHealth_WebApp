# MentalHealth_WepApp
# Mental Health Prediction Web App

## Overview
The **Mental Health Prediction Web App** is a multi-model machine learning application designed to assess and predict mental well-being based on various lifestyle, psychological, and work-related factors. The app predicts:

- **Stress Level (1-5 scale)**
- **Depression Risk (Yes/No)**
- **Burnout Status (Yes/No)**
- **Personalized Wellness Recommendations** (therapy, meditation, music, and relaxation needs)

The app performs **data analysis and visualization** to derive meaningful insights into mental health trends.

## Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Pickle
- **Machine Learning Models**: Various classification and regression algorithms

---

## Datasets
The project uses four datasets with different mental health indicators:

### 1️⃣ Stress Level Prediction Dataset (400 rows)
**Goal:** Predict stress levels (1-5) based on lifestyle, workload, and habits.

| Column Name      | Description                      | Values               |
|-----------------|--------------------------------|----------------------|
| age             | Age of the individual          | 18-60 years         |
| gender         | Gender                         | Male/Female/Other  |
| sleep_hours    | Average sleep per day         | 3-10 hours         |
| work_hours     | Daily work/study hours        | 4-14 hours         |
| exercise_freq  | Weekly exercise sessions      | 0-7 sessions       |
| screen_time    | Daily screen time             | 2-12 hours         |
| social_activity | Social interactions per week  | 0-10 interactions  |
| diet_quality   | Nutrition rating (1-5)        | 1 (Low) - 5 (High) |
| stress_level   | Predicted stress level        | 1-5 scale          |

### 2️⃣ Depression Risk Prediction Dataset (880 rows)
**Goal:** Predict whether a person is at risk of depression (Yes/No).

| Column Name      | Description                      | Values               |
|-----------------|--------------------------------|----------------------|
| age             | Age of the individual          | 18-60 years         |
| gender         | Gender                         | Male/Female/Other  |
| family_history | Family history of mental illness | Yes/No              |
| sleep_quality  | Sleep quality rating (1-5)    | 1-5 scale          |
| stress_level   | Stress level from previous model | 1-5 scale          |
| social_support | Level of emotional support (1-5) | 1-5 scale          |
| physical_activity | Weekly exercise sessions    | 0-7 sessions       |
| substance_use  | Smoking/Drinking (Yes/No)    | Yes/No              |
| depression_risk | Depression risk label        | Yes/No              |

### 3️⃣ Burnout Detection Dataset (750 rows)
**Goal:** Detect if an individual is experiencing burnout (Yes/No).

| Column Name     | Description                    | Values               |
|---------------|------------------------------|----------------------|
| age          | Age of the individual        | 18-60 years         |
| gender      | Gender                       | Male/Female/Other  |
| work_hours  | Daily work/study hours       | 4-16 hours         |
| overtime    | Extra work hours per week    | 0-20 hours         |
| work_satisfaction | Job satisfaction rating (1-5) | 1-5 scale          |
| stress_level | Stress level from previous model | 1-5 scale          |
| sleep_hours | Average sleep per day       | 3-10 hours         |
| mood_swings | Mood instability (1-5)       | 1-5 scale          |
| burnout     | Burnout detection label     | Yes/No              |

### 4️⃣ Wellness Recommendation Dataset (1000 rows)
**Goal:** Recommend relaxation techniques based on mental well-being.

| Column Name           | Description                          | Values               |
|---------------------|----------------------------------|----------------------|
| age                | Age of the individual           | 18-60 years         |
| gender            | Gender                          | Male/Female/Other  |
| stress_level      | Stress level from previous model | 1-5 scale          |
| burnout          | Burnout status from previous model | Yes/No              |
| depression_risk  | Depression risk from previous model | Yes/No              |
| meditation_needed | Meditation necessity rating (1-5) | 1-5 scale          |
| therapy_needed   | Therapy necessity rating (1-5)   | 1-5 scale          |
| music_needed     | Music therapy necessity rating (1-5) | 1-5 scale          |
| relaxation_techniques | Overall relaxation rating (1-5) | 1-5 scale          |

---

## Machine Learning Models & Evaluation
Each model was selected based on performance and dataset characteristics.

### 🔹 Stress Level Prediction Model
**Algorithm Used:** Random Forest Classifier  
**Performance:**
- **Accuracy:** 89%
- **Best Class Recall:** 91%-93%

### 🔹 Depression Risk Prediction Model
**Algorithm Used:** Logistic Regression  
**Performance:**
- **Accuracy:** 91%
- **Precision/Recall:** High across both classes

### 🔹 Burnout Detection Model
**Algorithm Used:** Decision Tree Classifier  
**Performance:**
- **Accuracy:** 100%
- **Perfect classification across all labels**

### 🔹 Wellness Recommendation Model
**Algorithm Used:** Multiple Linear Regression  
**Performance:**
- **Mean Absolute Error (MAE):** 0.48 - 0.84
- **Mean Squared Error (MSE):** Low values indicating good prediction accuracy

---

## Future Enhancements
- Improve model performance on low-recall classes
- Expand dataset to include more diverse samples
- Implement deep learning models for better predictions

---

## Acknowledgment
This project was developed by **Shivali Sharma** as a part of hands-on experience in data science, machine learning, and web app development. 🚀

