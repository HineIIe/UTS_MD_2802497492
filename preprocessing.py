def preprocess(df):

    df['academic_score'] = (
        df['ssc_percentage'] + 
        df['hsc_percentage'] + 
        df['degree_percentage'] + 
        (df['cgpa'] * 10)
    ) / 4

    df['total_experience'] = (
        df['internship_count'] + 
        df['live_projects'] + 
        (df['work_experience_months'] / 6)
    )

    df['skill_score'] = (
        df['technical_skill_score'] + 
        df['soft_skill_score']
    ) / 2

    df = df.drop(columns=['student_id'])

    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['extracurricular_activities'] = df['extracurricular_activities'].map({'Yes': 1, 'No': 0})

    return df