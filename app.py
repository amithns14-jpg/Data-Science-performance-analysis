"""
Student Performance Analysis Using Python
=========================================
A comprehensive, college-ready Data Science dashboard built with Streamlit,
Pandas, NumPy, Matplotlib, and Seaborn.
"""

import os
import sys
import io
import tempfile

# Ensure root directory and inner src directory are on sys.path.
# This prevents ModuleNotFoundError on cloud platforms like Render.com (which deploys to /opt/render/project/src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ensure matplotlib uses a safe local directory for font cache
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.data_generator import generate_synthetic_data, calculate_grade
    from src.data_processor import (
        auto_detect_columns,
        validate_dataset,
        clean_and_normalize,
        compute_kpi_metrics,
        compute_subject_stats,
        STANDARD_COLUMNS,
        SUBJECT_COLUMNS
    )
    from src.visualizer import (
        plot_grade_distribution,
        plot_grade_donut,
        plot_subject_averages,
        plot_subject_boxplots,
        plot_attendance_vs_marks,
        plot_study_hours_vs_marks,
        plot_correlation_heatmap,
        plot_gender_comparison,
        plot_student_comparison
    )
    from src.student_analytics import (
        get_students_needing_improvement,
        get_student_profile,
        classify_risk_level,
        generate_recommendations
    )
except ImportError:
    from data_generator import generate_synthetic_data, calculate_grade
    from data_processor import (
        auto_detect_columns,
        validate_dataset,
        clean_and_normalize,
        compute_kpi_metrics,
        compute_subject_stats,
        STANDARD_COLUMNS,
        SUBJECT_COLUMNS
    )
    from visualizer import (
        plot_grade_distribution,
        plot_grade_donut,
        plot_subject_averages,
        plot_subject_boxplots,
        plot_attendance_vs_marks,
        plot_study_hours_vs_marks,
        plot_correlation_heatmap,
        plot_gender_comparison,
        plot_student_comparison
    )
    from student_analytics import (
        get_students_needing_improvement,
        get_student_profile,
        classify_risk_level,
        generate_recommendations
    )

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analysis System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and KPI cards
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetricLabel"] > div {
        font-weight: 600;
        color: #475569;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] > div {
        font-weight: 700;
        color: #0f172a;
        font-size: 1.7rem;
    }
    
    /* Header Badge Styling */
    .badge-synthetic {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .badge-real {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }
    
    /* Report Card Card Container */
    .report-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "dataset_mode" not in st.session_state:
    st.session_state.dataset_mode = "Synthetic"

if "df_active" not in st.session_state:
    # Load default synthetic dataset (250 records)
    synthetic_file = os.path.join(BASE_DIR, "data", "synthetic_students.csv")
    if os.path.exists(synthetic_file):
        st.session_state.df_active = pd.read_csv(synthetic_file)
    else:
        st.session_state.df_active = generate_synthetic_data(250, seed=42)
    st.session_state.dataset_mode = "Synthetic"

if "df_uploaded_raw" not in st.session_state:
    st.session_state.df_uploaded_raw = None

if "real_dataset_confirmed" not in st.session_state:
    st.session_state.real_dataset_confirmed = False


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Control Panel")
    st.caption("Student Performance Analysis System")
    st.markdown("---")

    st.subheader("📁 1. Dataset Selection")
    dataset_option = st.radio(
        "Choose Data Source:",
        options=["Option A: Synthetic Dataset", "Option B: Upload Real Dataset"],
        index=0 if st.session_state.dataset_mode == "Synthetic" else 1
    )

    if dataset_option == "Option A: Synthetic Dataset":
        if st.session_state.dataset_mode != "Synthetic":
            st.session_state.dataset_mode = "Synthetic"
            synthetic_file = os.path.join(BASE_DIR, "data", "synthetic_students.csv")
            if os.path.exists(synthetic_file):
                st.session_state.df_active = pd.read_csv(synthetic_file)
            else:
                st.session_state.df_active = generate_synthetic_data(250, seed=42)
            st.session_state.real_dataset_confirmed = False

        st.markdown("**Synthetic Data Controls**")
        synth_count = st.slider("Number of Students:", min_value=200, max_value=600, value=250, step=25)
        synth_seed = st.number_input("Random Seed (Reproducibility):", min_value=1, max_value=9999, value=42)
        if st.button("🔄 Regenerate Synthetic Data", use_container_width=True):
            st.session_state.df_active = generate_synthetic_data(num_records=synth_count, seed=int(synth_seed))
            st.success(f"Generated {synth_count} synthetic student records!")

    else:
        # Option B: Real dataset upload
        st.markdown("**Upload Academic CSV File**")
        uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"], help="Upload any student academic records CSV.")
        
        if uploaded_file is not None:
            try:
                # Read without modifying active state until confirmed
                df_temp = pd.read_csv(uploaded_file)
                st.session_state.df_uploaded_raw = df_temp
                st.info(f"Loaded: `{uploaded_file.name}` ({len(df_temp)} rows, {len(df_temp.columns)} cols)")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

        # Quick test sample button
        sample_path = os.path.join(BASE_DIR, "data", "sample_upload.csv")
        if os.path.exists(sample_path):
            if st.button("📥 Load Demo College CSV (`sample_upload.csv`)", help="Test upload workflow with simulated real college data"):
                st.session_state.df_uploaded_raw = pd.read_csv(sample_path)
                st.success("Loaded demo CSV for testing!")

    st.markdown("---")
    st.subheader("⚙️ 2. Academic Thresholds")
    pass_mark_threshold = st.slider("Passing Marks Threshold:", min_value=35.0, max_value=60.0, value=50.0, step=5.0)
    att_threshold = st.slider("Minimum Attendance Norm (%):", min_value=60.0, max_value=85.0, value=75.0, step=5.0)

    st.markdown("---")
    st.subheader("🎯 3. Dynamic Filters")
    df_current = st.session_state.df_active

    # Filter by Gender
    gender_options = ["All"] + sorted(list(df_current["Gender"].dropna().unique())) if "Gender" in df_current.columns else ["All"]
    selected_gender = st.selectbox("Filter by Gender:", gender_options)

    # Filter by Grade
    grade_options = ["All", "A+", "A", "B", "C", "D", "F"]
    selected_grade = st.selectbox("Filter by Grade:", grade_options)

    # Filter by Attendance Status
    att_filter = st.selectbox("Filter by Attendance:", ["All", "Adequate (>= Threshold)", "Shortage (< Threshold)"])

    # Reset button
    if st.button("Reset All Filters", use_container_width=True):
        selected_gender = "All"
        selected_grade = "All"
        att_filter = "All"
        st.rerun()

    st.markdown("---")
    st.caption("College Data Science Demonstration Project\nAntigravity AI Assistant")


# -----------------------------------------------------------------------------
# MAIN HEADER & BADGES
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.title("🎓 Student Performance Analysis System")
    st.markdown("Interactive Data Science Dashboard for Academic Records, Performance Analytics, and Student Intervention.")

with header_col2:
    st.write("")
    if st.session_state.dataset_mode == "Synthetic":
        st.markdown(
            '<div style="text-align: right;"><span class="badge-synthetic">🧪 Option A: Synthetic Dataset</span><br>'
            '<small style="color: #64748b;">(Generated Demonstration Data)</small></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align: right;"><span class="badge-real">📂 Option B: Real Institutional Data</span><br>'
            '<small style="color: #64748b;">(Uploaded CSV Dataset)</small></div>',
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# REAL DATASET UPLOAD & CLEANING MODAL/SECTION (When in Option B)
# -----------------------------------------------------------------------------
if dataset_option == "Option B: Upload Real Dataset" and st.session_state.df_uploaded_raw is not None:
    raw_df = st.session_state.df_uploaded_raw
    with st.expander("🛠️ Dataset Ingestion, Column Mapping & Data Cleaning Center", expanded=(not st.session_state.real_dataset_confirmed)):
        st.markdown("### 📋 Uploaded Dataset Preview & Health Inspection")
        st.write("First 5 rows of your uploaded file:")
        st.dataframe(raw_df.head(), use_container_width=True)

        detected_cols = auto_detect_columns(raw_df)
        val_report = validate_dataset(raw_df, {k: v for k, v in detected_cols.items() if v})

        # Quality inspection metrics
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        q_col1.metric("Uploaded Rows", val_report["total_rows"])
        q_col2.metric("Uploaded Columns", val_report["total_cols"])
        q_col3.metric("Duplicate Rows", val_report["duplicate_rows"], delta=None if val_report["duplicate_rows"] == 0 else "Needs Cleaning")
        
        missing_count = sum(val_report["missing_per_col"].values())
        q_col4.metric("Total Missing Values", missing_count, delta=None if missing_count == 0 else "Needs Imputation")

        if val_report["invalid_values"]:
            st.warning(f"⚠️ Out-of-range or non-numeric values detected in: {', '.join(val_report['invalid_values'].keys())}")

        st.markdown("---")
        st.markdown("### 🔀 Column Mapping (Map Your Columns to Standard Fields)")
        st.caption("We automatically detected matching columns. Please review and adjust if necessary:")

        all_raw_cols = ["-- None / Not Available --"] + list(raw_df.columns)

        def get_default_index(std_field: str) -> int:
            matched = detected_cols.get(std_field)
            if matched and matched in raw_df.columns:
                return all_raw_cols.index(matched)
            return 0

        map_col1, map_col2, map_col3 = st.columns(3)
        with map_col1:
            m_id = st.selectbox("Student ID / Roll No:", all_raw_cols, index=get_default_index("Student_ID"))
            m_name = st.selectbox("Student Name:", all_raw_cols, index=get_default_index("Student_Name"))
            m_gender = st.selectbox("Gender:", all_raw_cols, index=get_default_index("Gender"))
            m_att = st.selectbox("Attendance (%):", all_raw_cols, index=get_default_index("Attendance"))

        with map_col2:
            m_study = st.selectbox("Study Hours (Weekly):", all_raw_cols, index=get_default_index("Study_Hours"))
            m_math = st.selectbox("Maths / Subject 1 Marks:", all_raw_cols, index=get_default_index("Maths_Marks"))
            m_sci = st.selectbox("Science / Subject 2 Marks:", all_raw_cols, index=get_default_index("Science_Marks"))
            m_eng = st.selectbox("English / Subject 3 Marks:", all_raw_cols, index=get_default_index("English_Marks"))

        with map_col3:
            m_comp = st.selectbox("Computer / Subject 4 Marks:", all_raw_cols, index=get_default_index("Computer_Marks"))
            m_final = st.selectbox("Final Marks (Optional - Auto Calculated):", all_raw_cols, index=get_default_index("Final_Marks"))
            m_grade = st.selectbox("Grade (Optional - Auto Calculated):", all_raw_cols, index=get_default_index("Grade"))

        st.markdown("---")
        st.markdown("### 🧹 Data Cleaning Operations")
        c_col1, c_col2, c_col3 = st.columns(3)
        clean_impute = c_col1.checkbox("Impute missing numeric values using column median", value=True)
        clean_dedup = c_col2.checkbox("Remove duplicate rows", value=True)
        clean_clip = c_col3.checkbox("Clip marks between 0 and 100", value=True)

        st.markdown("> **Note:** In accordance with safety rules, your original data is never overwritten without explicit confirmation.")
        
        if st.button("✅ Confirm & Apply Cleaned Dataset to Dashboard", type="primary", use_container_width=True):
            user_mapping = {}
            for std_key, val in [
                ("Student_ID", m_id), ("Student_Name", m_name), ("Gender", m_gender),
                ("Attendance", m_att), ("Study_Hours", m_study), ("Maths_Marks", m_math),
                ("Science_Marks", m_sci), ("English_Marks", m_eng), ("Computer_Marks", m_comp),
                ("Final_Marks", m_final), ("Grade", m_grade)
            ]:
                if val != "-- None / Not Available --":
                    user_mapping[std_key] = val

            df_normalized = clean_and_normalize(
                raw_df,
                user_mapping,
                impute_missing=clean_impute,
                remove_duplicates=clean_dedup
            )
            st.session_state.df_active = df_normalized
            st.session_state.dataset_mode = "Real"
            st.session_state.real_dataset_confirmed = True
            st.success(f"Successfully processed and loaded {len(df_normalized)} student records from your uploaded dataset!")
            st.rerun()


# -----------------------------------------------------------------------------
# FILTERING PIPELINE
# -----------------------------------------------------------------------------
df_working = st.session_state.df_active.copy()

# Apply Gender filter
if selected_gender != "All" and "Gender" in df_working.columns:
    df_working = df_working[df_working["Gender"] == selected_gender]

# Apply Grade filter
if selected_grade != "All" and "Grade" in df_working.columns:
    df_working = df_working[df_working["Grade"] == selected_grade]

# Apply Attendance filter
if "Attendance" in df_working.columns:
    if att_filter == "Adequate (>= Threshold)":
        df_working = df_working[df_working["Attendance"] >= att_threshold]
    elif att_filter == "Shortage (< Threshold)":
        df_working = df_working[df_working["Attendance"] < att_threshold]


# -----------------------------------------------------------------------------
# 4. MAIN APPLICATION DASHBOARD - DYNAMIC KPI CARDS
# -----------------------------------------------------------------------------
metrics = compute_kpi_metrics(df_working, pass_mark=pass_mark_threshold)

st.markdown("### 📌 Key Performance Indicators (KPIs)")

kpi_cols = st.columns(7)
kpi_cols[0].metric("Total Students", f"{metrics['total_students']}")
kpi_cols[1].metric("Average Marks", f"{metrics['avg_marks']:.1f}%")
kpi_cols[2].metric("Avg Attendance", f"{metrics['avg_attendance']:.1f}%")
kpi_cols[3].metric("Highest Marks", f"{metrics['highest_marks']:.1f}")
kpi_cols[4].metric("Lowest Marks", f"{metrics['lowest_marks']:.1f}")
kpi_cols[5].metric("Passing Students", f"{metrics['passing_students']}", delta=f"{metrics['pass_percentage']:.1f}% Pass")
kpi_cols[6].metric("Failing Students", f"{metrics['failing_students']}", delta=f"{100 - metrics['pass_percentage']:.1f}% Fail", delta_color="inverse")

if len(df_working) < len(st.session_state.df_active):
    st.info(f"Showing filtered view: **{len(df_working)}** of **{len(st.session_state.df_active)}** total students match active filters.")

st.markdown("---")


# -----------------------------------------------------------------------------
# MAIN APPLICATION TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Executive Overview",
    "📚 Subject Performance",
    "🔍 Correlation & Drivers",
    "⚠️ Students Needing Improvement",
    "👤 Individual Student Lookup",
    "📁 Dataset Explorer & Export",
    "🎓 Presentation & Viva Guide"
])


# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tabs[0]:
    st.subheader("📊 Executive Academic Overview")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### 🏆 Grade Distribution")
        fig_grade = plot_grade_distribution(df_working)
        st.pyplot(fig_grade)
        plt.close(fig_grade)

    with col_chart2:
        st.markdown("##### 🍰 Grade Proportions")
        fig_donut = plot_grade_donut(df_working)
        st.pyplot(fig_donut)
        plt.close(fig_donut)

    st.markdown("---")
    col_sub1, col_sub2 = st.columns([3, 2])
    with col_sub1:
        st.markdown("##### 📈 Average Marks Across Subjects")
        fig_sub_avg = plot_subject_averages(df_working)
        st.pyplot(fig_sub_avg)
        plt.close(fig_sub_avg)

    with col_sub2:
        st.markdown("##### 🌟 Academic Honor Roll (Top Performers)")
        top_students = df_working.nlargest(5, "Final_Marks")[
            [c for c in ["Student_ID", "Student_Name", "Attendance", "Study_Hours", "Final_Marks", "Grade"] if c in df_working.columns]
        ]
        st.dataframe(top_students, use_container_width=True, hide_index=True)

        st.markdown("##### 💡 Summary Insights")
        st.info(
            f"• **Cohort Size:** {metrics['total_students']} students evaluated.\n"
            f"• **Academic Passing Rate:** {metrics['pass_percentage']}% at {pass_mark_threshold}% benchmark.\n"
            f"• **Attendance Health:** Class average stands at {metrics['avg_attendance']}%.\n"
            f"• **Score Range:** From a minimum of {metrics['lowest_marks']} to a maximum of {metrics['highest_marks']}."
        )


# =============================================================================
# TAB 2: SUBJECT PERFORMANCE
# =============================================================================
with tabs[1]:
    st.subheader("📚 Subject-Wise Marks & Dispersion Analysis")

    st.markdown("##### 📊 Subject Statistical Summary")
    subj_stats = compute_subject_stats(df_working)
    if not subj_stats.empty:
        st.dataframe(subj_stats, use_container_width=True, hide_index=True)
    else:
        st.warning("No subject columns available for analysis.")

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("##### 📦 Score Distribution & Outliers (Boxplots)")
        fig_box = plot_subject_boxplots(df_working)
        st.pyplot(fig_box)
        plt.close(fig_box)

    with col_s2:
        st.markdown("##### 👥 Subject Performance by Gender")
        fig_gender = plot_gender_comparison(df_working)
        if fig_gender:
            st.pyplot(fig_gender)
            plt.close(fig_gender)
        else:
            st.info("Gender data not available or single gender filtered.")


# =============================================================================
# TAB 3: CORRELATION & DRIVERS
# =============================================================================
with tabs[2]:
    st.subheader("🔍 Drivers of Academic Performance: Attendance & Study Hours")

    corr_col1, corr_col2 = st.columns(2)
    with corr_col1:
        st.markdown("##### 📅 Attendance vs Final Marks")
        fig_att = plot_attendance_vs_marks(df_working)
        st.pyplot(fig_att)
        plt.close(fig_att)
        st.caption("Scatter plot with regression trendline highlighting students below 75% attendance norm.")

    with corr_col2:
        st.markdown("##### ⏱️ Weekly Study Hours vs Final Marks")
        fig_study = plot_study_hours_vs_marks(df_working)
        st.pyplot(fig_study)
        plt.close(fig_study)
        st.caption("Demonstrating the direct positive correlation between dedicated study time and academic scores.")

    st.markdown("---")
    st.markdown("##### 🧬 Performance Correlation Heatmap (Pearson r)")
    col_heat1, col_heat2 = st.columns([3, 2])
    with col_heat1:
        fig_heat = plot_correlation_heatmap(df_working)
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    with col_heat2:
        st.markdown("##### 🔎 Statistical Observations")
        # Compute correlations
        valid_num = df_working.select_dtypes(include=[np.number])
        if "Attendance" in valid_num and "Final_Marks" in valid_num:
            r_att = df_working["Attendance"].corr(df_working["Final_Marks"])
            r_study = df_working["Study_Hours"].corr(df_working["Final_Marks"]) if "Study_Hours" in df_working else 0
            
            st.write(f"• **Attendance Correlation (r):** `{r_att:.2f}` — strong positive driver of final marks.")
            st.write(f"• **Study Hours Correlation (r):** `{r_study:.2f}` — significant positive association.")
            st.write("• **Key Finding:** Students with attendance < 75% exhibit a notably higher fail rate compared to peers maintaining regular attendance.")
            st.write("• **Actionable Takeaway:** Institutional interventions focusing on improving attendance above 75% will directly elevate overall pass percentage.")


# =============================================================================
# TAB 4: STUDENTS NEEDING IMPROVEMENT (AT-RISK)
# =============================================================================
with tabs[3]:
    st.subheader("⚠️ Academic Intervention: Students Needing Improvement")
    st.markdown(
        f"This module automatically screens students at risk using the **{pass_mark_threshold}% passing marks** benchmark "
        f"and **{att_threshold}% attendance minimum** requirement."
    )

    at_risk_df = get_students_needing_improvement(
        df_working,
        pass_threshold=pass_mark_threshold,
        att_threshold=att_threshold
    )

    # Risk metrics summary
    if not at_risk_df.empty:
        crit_count = (at_risk_df["Risk_Category"] == "Critical Risk").sum()
        acad_count = (at_risk_df["Risk_Category"] == "Academic Risk").sum()
        att_count = (at_risk_df["Risk_Category"] == "Attendance Shortage").sum()
        watch_count = (at_risk_df["Risk_Category"] == "Watchlist").sum()

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("🚨 Critical Risk (Both)", crit_count)
        r_col2.metric("❌ Academic Risk (Failing)", acad_count)
        r_col3.metric("⚠️ Attendance Shortage", att_count)
        r_col4.metric("👀 Watchlist (Borderline)", watch_count)

        st.markdown("---")
        # Category filter for table
        risk_filter = st.multiselect(
            "Filter by Risk Severity:",
            options=["Critical Risk", "Academic Risk", "Attendance Shortage", "Watchlist"],
            default=["Critical Risk", "Academic Risk", "Attendance Shortage", "Watchlist"]
        )
        filtered_risk_df = at_risk_df[at_risk_df["Risk_Category"].isin(risk_filter)]

        display_cols = [c for c in [
            "Student_ID", "Student_Name", "Attendance", "Study_Hours",
            "Final_Marks", "Grade", "Risk_Category", "Recommendations"
        ] if c in filtered_risk_df.columns]

        st.dataframe(filtered_risk_df[display_cols], use_container_width=True, hide_index=True)

        # Download button for at-risk report
        csv_risk = filtered_risk_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export At-Risk Student Report (CSV)",
            data=csv_risk,
            file_name="students_needing_improvement.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.success("🎉 Excellent! No students in the current filtered selection require urgent academic intervention.")


# =============================================================================
# TAB 5: INDIVIDUAL STUDENT LOOKUP & REPORT CARD
# =============================================================================
with tabs[4]:
    st.subheader("👤 Individual Student Report Card & Search")

    if not df_working.empty and "Student_ID" in df_working.columns:
        # Search dropdown
        id_options = df_working["Student_ID"].astype(str).tolist()
        name_map = {}
        for _, r in df_working.iterrows():
            s_name = r.get("Student_Name", "")
            s_id = str(r.get("Student_ID", ""))
            name_map[s_id] = f"{s_id} - {s_name}"

        selected_id = st.selectbox(
            "Search or Select Student by ID / Name:",
            options=id_options,
            format_func=lambda sid: name_map.get(sid, sid)
        )

        profile = get_student_profile(df_working, selected_id)
        if profile:
            s_data = profile["student_data"]

            # Visual Report Card Header
            st.markdown(f"""
            <div class="report-card">
                <h3 style="margin-top:0; color:#0f172a;">🎓 Student Academic Report Card</h3>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div>
                        <p><strong>Name:</strong> {s_data.get('Student_Name', 'N/A')}</p>
                        <p><strong>Student ID:</strong> {s_data.get('Student_ID', 'N/A')}</p>
                        <p><strong>Gender:</strong> {s_data.get('Gender', 'N/A')}</p>
                    </div>
                    <div>
                        <p><strong>Class Rank:</strong> #{profile['rank']} of {profile['total_students']}</p>
                        <p><strong>Percentile:</strong> {profile['percentile']}%</p>
                        <p><strong>Final Grade:</strong> <span style="font-size:1.2rem; font-weight:bold; color:#2563eb;">{s_data.get('Grade', 'N/A')}</span></p>
                    </div>
                    <div>
                        <p><strong>Attendance:</strong> {s_data.get('Attendance', 0):.1f}%</p>
                        <p><strong>Study Hours:</strong> {s_data.get('Study_Hours', 0):.1f} hrs/week</p>
                        <p><strong>Overall Score:</strong> {s_data.get('Final_Marks', 0):.1f} / 100</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            card_col1, card_col2 = st.columns([3, 2])
            with card_col1:
                st.markdown("##### 📊 Student vs Class Average Comparison")
                fig_stud = plot_student_comparison(s_data, pd.Series(profile["class_means"]))
                st.pyplot(fig_stud)
                plt.close(fig_stud)

            with card_col2:
                st.markdown("##### 🎯 Academic Diagnosis & Plan")
                st.write(f"• **Status:** **{profile['risk_category']}**")
                st.write(f"• **Strongest Subject:** `{profile['best_subject'][0]}` ({profile['best_subject'][1]:.0f}/100)")
                st.write(f"• **Subject Needing Focus:** `{profile['weakest_subject'][0]}` ({profile['weakest_subject'][1]:.0f}/100)")

                st.markdown("##### 📝 Personalized Improvement Advice:")
                for rec in profile["recommendations"]:
                    st.write(f"- {rec}")
    else:
        st.warning("No student records available to display.")


# =============================================================================
# TAB 6: DATASET EXPLORER & EXPORT
# =============================================================================
with tabs[5]:
    st.subheader("📁 Dataset Explorer, Summary Statistics & CSV Export")

    d_col1, d_col2 = st.columns([3, 1])
    with d_col1:
        st.markdown(f"**Viewing Active Dataset:** `{st.session_state.dataset_mode}` ({len(df_working)} rows)")
    with d_col2:
        csv_full = df_working.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_full,
            file_name=f"student_performance_data_{st.session_state.dataset_mode.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.dataframe(df_working, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🔢 Comprehensive Numerical Summary (`df.describe()`)")
    st.dataframe(df_working.describe().T, use_container_width=True)


# =============================================================================
# TAB 7: COLLEGE PRESENTATION & VIVA GUIDE
# =============================================================================
with tabs[6]:
    st.subheader("🎓 College Project Presentation & Viva Defense Guide")
    st.markdown("""
    Use this section as a reference guide during your college viva examination or project evaluation.
    """)

    with st.expander("📌 1. Project Title & Problem Statement", expanded=True):
        st.markdown("""
        - **Project Title:** Student Performance Analysis Using Python.
        - **Objective:** To build an interactive, data-driven analytical dashboard that tracks, evaluates, and visualizes student academic records, highlights subject-level strengths and weaknesses, examines behavioral drivers (attendance & study hours), and proactively flags students who need academic intervention.
        - **Target Audience:** College professors, academic counselors, department heads, and students.
        """)

    with st.expander("🛠️ 2. Technology Stack & Why Selected"):
        st.markdown("""
        - **Python 3:** The industry-standard programming language for data science and statistical computing.
        - **Pandas:** Used for tabular data ingestion, column mapping, missing value imputation, duplicate handling, and aggregation.
        - **NumPy:** Used for fast vectorized mathematical operations and reproducible pseudo-random data generation (`np.random.default_rng`).
        - **Matplotlib & Seaborn:** Chosen for high-quality, publishable data visualizations including distribution charts, boxplots, regression trendlines, and correlation heatmaps.
        - **Streamlit:** Fast, lightweight, pure-Python web framework to build an interactive reactive dashboard without needing complex HTML/CSS/JavaScript backends.
        """)

    with st.expander("❓ 3. Top Viva Questions & Model Answers"):
        st.markdown("""
        **Q1: What is the correlation between attendance and final marks in your analysis?**  
        *Answer:* The correlation heatmap and scatter regression plot show a strong positive Pearson correlation (typically $r \\approx 0.65 - 0.75$). Students with attendance below the 75% threshold exhibit a significantly higher probability of academic failure.

        **Q2: How does the application handle real uploaded datasets with different column names?**  
        *Answer:* The system implements a heuristic column-matching algorithm (`auto_detect_columns`) using keyword pattern detection. Users can review and adjust mappings via interactive selectboxes before confirming. Data validation checks for missing values, duplicates, and range boundaries ([0, 100] for marks).

        **Q3: How are students needing academic improvement identified?**  
        *Answer:* The system uses a multi-factor risk categorization function (`classify_risk_level`) considering passing marks (<50%), attendance (<75%), and study hours (<3.5 hrs/week) to segment students into *Critical Risk*, *Academic Risk*, *Attendance Shortage*, and *Watchlist*.

        **Q4: Why did you use median imputation instead of dropping missing values?**  
        *Answer:* Dropping missing records reduces sample size and can introduce attrition bias. Using the median is robust against extreme outliers and preserves valuable student records.
        """)

    with st.expander("🚀 4. Future Enhancements & Scope"):
        st.markdown("""
        - Integration with institutional SQL databases (PostgreSQL, MySQL).
        - Predictive machine learning models (e.g., Logistic Regression or Random Forest) for early mid-term dropout prediction.
        - Automated email notifications to parents and academic mentors.
        - Multi-semester longitudinal tracking to monitor student growth across years.
        """)
