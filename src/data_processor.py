"""
Data Processing, Validation, and Cleaning Module
Handles data ingestion, validation, cleaning, and metric computation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


STANDARD_COLUMNS = [
    "Student_ID",
    "Student_Name",
    "Gender",
    "Attendance",
    "Study_Hours",
    "Maths_Marks",
    "Science_Marks",
    "English_Marks",
    "Computer_Marks",
    "Final_Marks",
    "Grade"
]

SUBJECT_COLUMNS = [
    "Maths_Marks",
    "Science_Marks",
    "English_Marks",
    "Computer_Marks"
]


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Heuristically map uploaded DataFrame column names to standard fields.
    """
    detected: Dict[str, Optional[str]] = {}
    col_names = list(df.columns)
    col_lower_map = {str(c).strip().lower(): c for c in col_names}

    keywords = {
        "Student_ID": ["id", "student_id", "roll", "roll_no", "rollno", "studentid"],
        "Student_Name": ["name", "student_name", "studentname", "full_name", "fullname"],
        "Gender": ["gender", "sex"],
        "Attendance": ["attendance", "attendance_pct", "attendance_percentage", "attendance%"],
        "Study_Hours": ["study_hours", "study_hour", "hours", "studyhours", "study_time"],
        "Maths_Marks": ["maths", "math", "maths_marks", "mathematics", "math_score"],
        "Science_Marks": ["science", "science_marks", "sci", "science_score"],
        "English_Marks": ["english", "english_marks", "eng", "english_score"],
        "Computer_Marks": ["computer", "computer_marks", "computers", "cs", "cs_marks", "it"],
        "Final_Marks": ["final_marks", "final", "total", "total_marks", "percentage", "average", "avg_marks"],
        "Grade": ["grade", "letter_grade", "final_grade"]
    }

    used_cols = set()

    for std_field, patterns in keywords.items():
        matched = None
        # Exact or close match
        for p in patterns:
            if p in col_lower_map and col_lower_map[p] not in used_cols:
                matched = col_lower_map[p]
                break

        # Substring search if not exact
        if not matched:
            for p in patterns:
                for low, orig in col_lower_map.items():
                    if orig not in used_cols and (p in low or low in p):
                        matched = orig
                        break
                if matched:
                    break

        detected[std_field] = matched
        if matched:
            used_cols.add(matched)

    return detected


def validate_dataset(df: pd.DataFrame, column_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Inspect dataset quality: missing values, duplicates, out-of-range marks, data types.
    """
    report = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_per_col": df.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "invalid_values": {}
    }

    # Check value ranges for mapped numeric columns
    numeric_checks = {
        "Attendance": (0.0, 100.0),
        "Study_Hours": (0.0, 24.0),
        "Maths_Marks": (0.0, 100.0),
        "Science_Marks": (0.0, 100.0),
        "English_Marks": (0.0, 100.0),
        "Computer_Marks": (0.0, 100.0),
        "Final_Marks": (0.0, 100.0)
    }

    for std_col, (min_v, max_v) in numeric_checks.items():
        orig_col = column_map.get(std_col)
        if orig_col and orig_col in df.columns:
            s = pd.to_numeric(df[orig_col], errors="coerce")
            invalid_mask = (s < min_v) | (s > max_v)
            num_invalid = int(invalid_mask.sum())
            nan_count = int(s.isnull().sum() - df[orig_col].isnull().sum())
            if num_invalid > 0 or nan_count > 0:
                report["invalid_values"][orig_col] = {
                    "expected_range": f"{min_v} - {max_v}",
                    "out_of_range_count": num_invalid,
                    "non_numeric_count": nan_count
                }

    return report


def assign_grade(score: float) -> str:
    """Helper to assign grade string from score."""
    if pd.isna(score):
        return "N/A"
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


def clean_and_normalize(
    df: pd.DataFrame,
    column_map: Dict[str, str],
    impute_missing: bool = True,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Standardize mapped dataset into canonical schema with optional imputation and deduplication.
    """
    df_clean = df.copy()

    if remove_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Invert mapping to rename columns
    inv_map = {orig: std for std, orig in column_map.items() if orig}
    # Select columns that are mapped
    selected_cols = list(inv_map.keys())
    standard_df = df_clean[selected_cols].rename(columns=inv_map)

    # Ensure required numeric fields are float/int
    numeric_cols = [
        "Attendance", "Study_Hours", "Maths_Marks",
        "Science_Marks", "English_Marks", "Computer_Marks", "Final_Marks"
    ]

    for col in numeric_cols:
        if col in standard_df.columns:
            standard_df[col] = pd.to_numeric(standard_df[col], errors="coerce")
            if impute_missing:
                med_val = standard_df[col].median()
                if pd.isna(med_val):
                    med_val = 50.0
                standard_df[col] = standard_df[col].fillna(med_val)
            # Clip bounds
            if col in ["Attendance", "Maths_Marks", "Science_Marks", "English_Marks", "Computer_Marks", "Final_Marks"]:
                standard_df[col] = standard_df[col].clip(0.0, 100.0)
            elif col == "Study_Hours":
                standard_df[col] = standard_df[col].clip(0.0, 24.0)

    # Impute categorical
    if "Gender" in standard_df.columns and impute_missing:
        mode_gender = standard_df["Gender"].mode()
        fallback = mode_gender.iloc[0] if not mode_gender.empty else "Unspecified"
        standard_df["Gender"] = standard_df["Gender"].fillna(fallback)

    # Generate Student_ID if missing
    if "Student_ID" not in standard_df.columns or standard_df["Student_ID"].isnull().all():
        standard_df["Student_ID"] = [f"STU{1001 + i}" for i in range(len(standard_df))]
    else:
        standard_df["Student_ID"] = standard_df["Student_ID"].astype(str)

    # Generate Student_Name if missing
    if "Student_Name" not in standard_df.columns or standard_df["Student_Name"].isnull().all():
        standard_df["Student_Name"] = [f"Student_{i+1}" for i in range(len(standard_df))]
    else:
        standard_df["Student_Name"] = standard_df["Student_Name"].astype(str)

    # Derive Final_Marks if missing or not mapped
    present_subjects = [c for c in SUBJECT_COLUMNS if c in standard_df.columns]
    if "Final_Marks" not in standard_df.columns:
        if present_subjects:
            standard_df["Final_Marks"] = standard_df[present_subjects].mean(axis=1).round(1)
        else:
            standard_df["Final_Marks"] = 0.0

    # Derive Grade if missing or not mapped
    if "Grade" not in standard_df.columns:
        standard_df["Grade"] = standard_df["Final_Marks"].apply(assign_grade)

    return standard_df


def compute_kpi_metrics(df: pd.DataFrame, pass_mark: float = 50.0) -> Dict[str, Any]:
    """
    Compute essential KPI metrics required for the dashboard.
    """
    total_students = len(df)
    if total_students == 0:
        return {
            "total_students": 0,
            "avg_marks": 0.0,
            "avg_attendance": 0.0,
            "highest_marks": 0.0,
            "lowest_marks": 0.0,
            "passing_students": 0,
            "failing_students": 0,
            "pass_percentage": 0.0
        }

    final_marks = df["Final_Marks"] if "Final_Marks" in df.columns else pd.Series([0])
    attendance = df["Attendance"] if "Attendance" in df.columns else pd.Series([0])

    avg_marks = float(round(final_marks.mean(), 1))
    avg_attendance = float(round(attendance.mean(), 1))
    highest_marks = float(round(final_marks.max(), 1))
    lowest_marks = float(round(final_marks.min(), 1))

    passing_students = int((final_marks >= pass_mark).sum())
    failing_students = int(total_students - passing_students)
    pass_pct = round((passing_students / total_students) * 100.0, 1)

    return {
        "total_students": total_students,
        "avg_marks": avg_marks,
        "avg_attendance": avg_attendance,
        "highest_marks": highest_marks,
        "lowest_marks": lowest_marks,
        "passing_students": passing_students,
        "failing_students": failing_students,
        "pass_percentage": pass_pct
    }


def compute_subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate statistical summary for available subject marks.
    """
    available_subjects = [col for col in SUBJECT_COLUMNS if col in df.columns]
    if not available_subjects:
        return pd.DataFrame()

    stats = []
    for col in available_subjects:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        name_clean = col.replace("_Marks", "")
        stats.append({
            "Subject": name_clean,
            "Average": round(series.mean(), 1),
            "Highest": round(series.max(), 1),
            "Lowest": round(series.min(), 1),
            "Std Dev": round(series.std(), 1),
            "Median": round(series.median(), 1),
            "Pass Rate (>50) %": round((series >= 50).mean() * 100, 1),
            "Distinction (>75) %": round((series >= 75).mean() * 100, 1)
        })

    return pd.DataFrame(stats)
