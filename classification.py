import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_classification(df):

    X = df[[
        'gender',
        'entrance_exam_score',
        'certifications',
        'attendance_percentage',
        'backlogs',
        'extracurricular_activities',
        'academic_score',
        'total_experience',
        'skill_score'
    ]]

    y = df['placement_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)

    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.7).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return pipeline, metrics