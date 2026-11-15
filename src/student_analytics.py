"""
Student Performance Analytics & Academic Intervention Module
Identifies at-risk students, generates actionable recommendations, and computes individual profiles.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def classify_risk_level(row: pd.Series, pass_threshold: float = 50.0, att_threshold: float = 75.0) -> str:
    """
    Determine risk level based on marks, attendance, and study hours.
    Levels: Critical, Moderate, Watchlist, Normal.
    """
    final_marks = float(row.get("Final_Marks", 0))
    attendance = float(row.get("Attendance", 0))
    study_hours = float(row.get("Study_Hours", 0))

    marks_failed = final_marks < pass_threshold
    att_shortage = attendance < att_threshold

    if marks_failed and att_shortage:
        return "Critical Risk"
    elif marks_failed:
        return "Academic Risk"
    elif att_shortage:
        return "Attendance Shortage"
    elif final_marks < (pass_threshold + 10) or study_hours < 3.5:
        return "Watchlist"
    else:
        return "Good Standing"


def generate_recommendations(row: pd.Series) -> List[str]:
    """
    Generate personalized academic improvement recommendations for a student.
    """
    recs = []
    final_marks = float(row.get("Final_Marks", 0))
    attendance = float(row.get("Attendance", 0))
    study_hours = float(row.get("Study_Hours", 0))

    # Attendance recommendation
    if attendance < 65:
        recs.append("Severe attendance shortage (<65%). Mandatory parent-teacher meeting & attendance recovery plan required.")
    elif attendance < 75:
        recs.append("Attendance is below minimum 75% college norm. Regular class attendance must be prioritized.")

    # Study hours recommendation
    if study_hours < 3.5:
        recs.append(f"Dedicated study time ({study_hours} hrs/week) is too low. Recommend setting a routine of at least 6-8 hrs/week.")
    elif study_hours < 5.0:
        recs.append("Increase weekly revision time by 2-3 hours focusing on weaker subjects.")

    # Subject-specific weaknesses
    subjects = [("Maths", "Maths_Marks"), ("Science", "Science_Marks"),
                ("English", "English_Marks"), ("Computer", "Computer_Marks")]

    weak_subjects = []
    for s_name, col in subjects:
        if col in row and pd.notnull(row[col]):
            score = float(row[col])
            if score < 50:
                weak_subjects.append(f"{s_name} ({score:.0f}/100)")

    if weak_subjects:
        recs.append(f"Failing scores in: {', '.join(weak_subjects)}. Remedial tutoring and concept-clearing sessions needed.")
    elif final_marks < 60:
        recs.append("Performance is on the border. Practice previous years' exam papers and weekly problem sets.")
    else:
        if not recs:
            recs.append("Strong academic progress! Maintain consistency and engage in peer tutoring or advanced problem solving.")

    return recs


def get_students_needing_improvement(
    df: pd.DataFrame,
    pass_threshold: float = 50.0,
    att_threshold: float = 75.0
) -> pd.DataFrame:
    """
    Filter and compile students who need academic intervention or improvement.
    """
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy["Risk_Category"] = df_copy.apply(
        lambda r: classify_risk_level(r, pass_threshold, att_threshold), axis=1
    )
    df_copy["Recommendations"] = df_copy.apply(
        lambda r: "; ".join(generate_recommendations(r)), axis=1
    )

    # Filter out students in "Good Standing"
    at_risk_df = df_copy[df_copy["Risk_Category"] != "Good Standing"].copy()

    # Sort so Critical Risk and Academic Risk appear on top, then by Final_Marks ascending
    priority_order = {"Critical Risk": 1, "Academic Risk": 2, "Attendance Shortage": 3, "Watchlist": 4}
    at_risk_df["Priority"] = at_risk_df["Risk_Category"].map(priority_order).fillna(5)
    at_risk_df = at_risk_df.sort_values(by=["Priority", "Final_Marks"], ascending=[True, True])
    at_risk_df = at_risk_df.drop(columns=["Priority"])

    return at_risk_df


def get_student_profile(df: pd.DataFrame, student_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve full comprehensive academic profile for an individual student.
    """
    match = df[df["Student_ID"].astype(str) == str(student_id)]
    if match.empty:
        return None

    student = match.iloc[0]
    total_students = len(df)

    # Calculate rank and percentile based on Final_Marks
    rank = int((df["Final_Marks"] > student["Final_Marks"]).sum() + 1)
    percentile = round(((total_students - rank + 1) / total_students) * 100, 1)

    # Class averages
    subj_cols = ["Maths_Marks", "Science_Marks", "English_Marks", "Computer_Marks"]
    class_means = {col: round(df[col].mean(), 1) for col in subj_cols if col in df.columns}

    # Best and weakest subject
    subj_scores = {}
    for col in subj_cols:
        if col in student and pd.notnull(student[col]):
            subj_scores[col.replace("_Marks", "")] = float(student[col])

    best_subject = max(subj_scores.items(), key=lambda x: x[1]) if subj_scores else ("N/A", 0)
    weakest_subject = min(subj_scores.items(), key=lambda x: x[1]) if subj_scores else ("N/A", 0)

    recs = generate_recommendations(student)
    risk = classify_risk_level(student)

    return {
        "student_data": student,
        "rank": rank,
        "total_students": total_students,
        "percentile": percentile,
        "best_subject": best_subject,
        "weakest_subject": weakest_subject,
        "recommendations": recs,
        "risk_category": risk,
        "class_means": class_means
    }
