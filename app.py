import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Models
# =========================
clf_model = joblib.load("models/clf_model.pkl")
reg_model = joblib.load("models/reg_model.pkl")

st.set_page_config(page_title="Student Placement Predictor", layout="wide")

st.title("🎓 Student Placement & Salary Predictor")
st.markdown("Predict whether a student gets placed and estimate their salary.")

# =========================
# INPUT SECTION (BETTER UI)
# =========================
st.sidebar.header("📥 Input Student Data")

# basic info
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
extracurricular = st.sidebar.selectbox("Extracurricular", ["Yes", "No"])

# academic
st.sidebar.subheader("📚 Academic")
ssc = st.sidebar.slider("SSC %", 50, 100, 70)
hsc = st.sidebar.slider("HSC %", 50, 100, 70)
degree = st.sidebar.slider("Degree %", 55, 100, 70)
cgpa = st.sidebar.slider("CGPA", 5.5, 10.0, 7.5)

# skills
st.sidebar.subheader("💡 Skills")
technical = st.sidebar.slider("Technical Skill", 40, 100, 70)
soft = st.sidebar.slider("Soft Skill", 40, 100, 70)

# experience
st.sidebar.subheader("💼 Experience")
internship = st.sidebar.slider("Internships", 0, 4, 1)
projects = st.sidebar.slider("Live Projects", 0, 5, 2)
work_exp = st.sidebar.slider("Work Experience (months)", 0, 24, 6)

# other
st.sidebar.subheader("📊 Other")
entrance_exam_score = st.sidebar.slider("Entrance Exam Score", 40, 100, 70)
certifications = st.sidebar.slider("Certifications", 0, 5, 2)
attendance = st.sidebar.slider("Attendance %", 60, 100, 80)
backlogs = st.sidebar.slider("Backlogs", 0, 5, 0)

# =========================
# Feature Engineering
# =========================
gender = 1 if gender == "Male" else 0
extracurricular = 1 if extracurricular == "Yes" else 0

academic_score = (ssc + hsc + degree + (cgpa * 10)) / 4
total_experience = internship + projects + (work_exp / 6)
skill_score = (technical + soft) / 2

# =========================
# Display Engineered Features
# =========================
st.subheader("🧠 Computed Scores")

col1, col2, col3 = st.columns(3)
col1.metric("Academic Score", f"{academic_score:.2f}")
col2.metric("Skill Score", f"{skill_score:.2f}")
col3.metric("Total Experience", f"{total_experience:.2f}")

# =========================
# Prepare Classification Input
# =========================
input_data = pd.DataFrame([{
    'gender': gender,
    'entrance_exam_score': entrance_exam_score,
    'certifications': certifications,
    'attendance_percentage': attendance,
    'backlogs': backlogs,
    'extracurricular_activities': extracurricular,
    'academic_score': academic_score,
    'total_experience': total_experience,
    'skill_score': skill_score
}])

# =========================
# Prediction
# =========================
if st.button("🔮 Predict", use_container_width=True):

    try:
        # Classification
        placement_prob = clf_model.predict_proba(input_data)[0][1]
        placement = 1 if placement_prob > 0.7 else 0

        st.subheader("📌 Placement Result")

        # Confidence bar
        st.progress(int(placement_prob * 100))

        if placement == 1:
            st.success(f"Placed ✅ (Confidence: {placement_prob:.2f})")
        else:
            st.error(f"Not Placed ❌ (Confidence: {placement_prob:.2f})")

        # Regression 
        if placement == 1:

            reg_input = pd.DataFrame([{
                'gender': gender,
                'degree_percentage': degree,
                'cgpa': cgpa,
                'entrance_exam_score': entrance_exam_score,
                'technical_skill_score': technical,
                'soft_skill_score': soft,
                'internship_count': internship,
                'live_projects': projects,
                'work_experience_months': work_exp,
                'certifications': certifications,
                'backlogs': backlogs,
                'extracurricular_activities': extracurricular,
                'academic_score': academic_score,
                'total_experience': total_experience,
                'skill_score': skill_score
            }])

            salary_log = reg_model.predict(reg_input)[0]
            salary = np.expm1(salary_log)

            st.subheader("💰 Estimated Salary")

            colA, colB = st.columns(2)
            colA.metric("Predicted Salary (LPA)", f"{salary:.2f}")
            colB.metric("Log Value", f"{salary_log:.2f}")

        else:
            st.info("Salary prediction only available if placed.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")