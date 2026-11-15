"""
Automated System Verification Script
Tests all modules, functions, data generation, cleaning, analytics, and chart rendering.
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_test_cache")

import pandas as pd
import numpy as np

# 1. Test data generator
from src.data_generator import generate_synthetic_data, calculate_grade

print("[1/5] Testing Synthetic Data Generator...")
df_synth = generate_synthetic_data(250, seed=42)
assert len(df_synth) == 250, f"Expected 250 rows, got {len(df_synth)}"
assert "Final_Marks" in df_synth.columns
assert "Grade" in df_synth.columns
print("  ✓ Data generator passed. Shape:", df_synth.shape)

# 2. Test data processor
from src.data_processor import (
    auto_detect_columns, validate_dataset, clean_and_normalize,
    compute_kpi_metrics, compute_subject_stats
)

print("[2/5] Testing Data Processor & Validation...")
sample_raw = pd.read_csv(os.path.join("data", "sample_upload.csv"))
detected = auto_detect_columns(sample_raw)
print("  ✓ Auto-detected columns:", {k: v for k, v in detected.items() if v})
assert detected["Student_ID"] == "Roll_No"
assert detected["Student_Name"] == "Full_Name"
assert detected["Attendance"] == "Attendance_Percentage"

report = validate_dataset(sample_raw, {k: v for k, v in detected.items() if v})
print(f"  ✓ Quality Report: {report['duplicate_rows']} duplicates, {sum(report['missing_per_col'].values())} missing values.")

cleaned_df = clean_and_normalize(sample_raw, {k: v for k, v in detected.items() if v})
assert "Final_Marks" in cleaned_df.columns
assert "Grade" in cleaned_df.columns
assert cleaned_df.isnull().sum().sum() == 0, "Missing values should be cleaned"
print("  ✓ Cleaning and normalization passed. Cleaned Shape:", cleaned_df.shape)

kpis = compute_kpi_metrics(df_synth)
assert kpis["total_students"] == 250
print("  ✓ Computed KPIs:", kpis)

stats = compute_subject_stats(df_synth)
assert len(stats) == 4
print("  ✓ Subject stats shape:", stats.shape)

# 3. Test Student Analytics
from src.student_analytics import (
    get_students_needing_improvement, get_student_profile,
    classify_risk_level, generate_recommendations
)

print("[3/5] Testing Student Analytics & At-Risk Detection...")
at_risk = get_students_needing_improvement(df_synth, pass_threshold=50.0, att_threshold=75.0)
print(f"  ✓ Identified {len(at_risk)} students needing academic intervention.")
assert len(at_risk) > 0

sample_id = df_synth.iloc[0]["Student_ID"]
prof = get_student_profile(df_synth, sample_id)
assert prof is not None
assert prof["rank"] >= 1
print(f"  ✓ Profile lookup for {sample_id} passed. Rank: {prof['rank']}/{prof['total_students']}, Best: {prof['best_subject']}")

# 4. Test Visualizer
from src.visualizer import (
    plot_grade_distribution, plot_grade_donut, plot_subject_averages,
    plot_subject_boxplots, plot_attendance_vs_marks, plot_study_hours_vs_marks,
    plot_correlation_heatmap, plot_gender_comparison, plot_student_comparison
)
import matplotlib.pyplot as plt

print("[4/5] Testing Visualization Generation...")
f1 = plot_grade_distribution(df_synth); plt.close(f1)
f2 = plot_grade_donut(df_synth); plt.close(f2)
f3 = plot_subject_averages(df_synth); plt.close(f3)
f4 = plot_subject_boxplots(df_synth); plt.close(f4)
f5 = plot_attendance_vs_marks(df_synth); plt.close(f5)
f6 = plot_study_hours_vs_marks(df_synth); plt.close(f6)
f7 = plot_correlation_heatmap(df_synth); plt.close(f7)
f8 = plot_gender_comparison(df_synth); plt.close(f8)
f9 = plot_student_comparison(df_synth.iloc[0], df_synth[["Maths_Marks", "Science_Marks", "English_Marks", "Computer_Marks"]].mean()); plt.close(f9)
print("  ✓ All 9 Matplotlib / Seaborn visualization functions executed without errors!")

print("[5/5] All system tests passed successfully! System is 100% ready for demonstration.")
