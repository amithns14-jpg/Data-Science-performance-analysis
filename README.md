# 🎓 Student Performance Analysis System Using Python

A complete, interactive Data Science and Analytics project designed for academic evaluation, performance diagnosis, and college demonstrations. Built with **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Streamlit**.

---

## 🌟 Key Highlights & Capabilities

- **Dual Dataset Support:**
  - **Option A (Synthetic Data):** Pre-generated 250 realistic student records with reproducible seeds (`np.random.default_rng`), realistically simulating correlations between attendance, study hours, and academic marks.
  - **Option B (Real Data Ingestion):** Upload any custom CSV file. Includes heuristic column auto-detection, missing value detection, duplicate row auditing, out-of-range value validation, median imputation, and flexible column mapping.
- **Dynamic KPI Dashboard:**
  - Real-time dynamic metrics: Total Students, Class Average Marks, Average Attendance, Highest Mark, Lowest Mark, Students Passing, and Students Failing (with dynamic pass/fail percentages).
- **Multi-Subject Analytics & Dispersion:**
  - Statistical summaries (Mean, Median, Standard Deviation, Min, Max, Distinction %, Pass Rate %).
  - Boxplots with jitter points for spread and outlier detection across Maths, Science, English, and Computer.
- **Academic Drivers & Correlation Analysis:**
  - Scatter plots with regression trendlines for **Attendance vs. Final Marks** (with 75% attendance threshold benchmark) and **Study Hours vs. Final Marks**.
  - Pearson correlation heatmap ($r$) revealing academic performance drivers.
- **Academic Intervention & At-Risk Profiling:**
  - Multi-tier student risk stratification: *Critical Risk*, *Academic Risk*, *Attendance Shortage*, and *Watchlist*.
  - Automated, actionable academic recommendations per student.
  - One-click CSV export of at-risk student lists for faculty intervention.
- **Individual Student Report Card & Search:**
  - Quick search by Student ID or Name.
  - Digital report card featuring class rank, cohort percentile, best subject, weakest subject, comparison against class averages, and tailored feedback.
- **Interactive Dataset Explorer & Export:**
  - Searchable, sortable data table, statistical summaries (`df.describe()`), and one-click export of cleaned datasets.
- **Presentation & Viva Voce Guide:**
  - Dedicated interactive tab with sample viva questions, methodology defense, and architectural explanations.

---

## 📂 Project Architecture

```
data-performance-analysis/
├── app.py                      # Main Streamlit interactive dashboard
├── requirements.txt            # Python package dependencies
├── run_app.bat                 # 1-click Windows launcher script
├── test_system.py              # Automated system test suite (5 verification phases)
├── generate_sample_csvs.py     # Generator for sample & synthetic CSV files
├── README.md                   # Complete documentation & college viva guide
│
├── src/                        # Modular Python backend packages
│   ├── __init__.py
│   ├── data_generator.py       # Reproducible synthetic student data generator (>=200 records)
│   ├── data_processor.py       # Cleaning, validation, column mapping & KPI metrics
│   ├── visualizer.py           # Matplotlib & Seaborn chart generator functions
│   └── student_analytics.py    # At-risk screening, risk scoring & student profiling
│
└── data/                       # Dataset directory
    ├── synthetic_students.csv  # 250 pre-generated synthetic records (reproducible seed=42)
    └── sample_upload.csv       # Test dataset with custom column headers, missing values & duplicates
```

---

## 🛠️ Technology Stack

| Technology | Role |
|---|---|
| **Python 3.13** | Core programming language |
| **Pandas** | Tabular data processing, column mapping, missing value imputation, and aggregations |
| **NumPy** | Vectorized statistical computations and reproducible pseudo-random data modeling |
| **Matplotlib** | Figure canvas, custom styling, annotations, and multi-axis chart rendering |
| **Seaborn** | High-level statistical visualizations (boxplots, regression plots, correlation heatmaps) |
| **Streamlit** | Interactive reactive web dashboard, widgets, and dynamic KPI metric cards |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
To verify that all modules and visualizations function properly:
```bash
python test_system.py
```

### 4. Launch the Dashboard
**Option A — Using the Windows launcher:**
Double-click `run_app.bat` or run:
```cmd
run_app.bat
```

**Option B — Using Streamlit command:**
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Demonstration Workflow for College Presentation

1. **Overview & KPI Demonstration:**
   - Launch the dashboard and show the 7 KPI cards.
   - Adjust the **Passing Marks Threshold** (e.g. from 50% to 40%) in the sidebar and observe all metrics and pass/fail counts dynamically update.
2. **Subject Performance & Spread:**
   - Navigate to **📚 Subject Performance** to present the boxplots and statistical summary table (showing standard deviation and distinction rates).
3. **Correlation & Behavioral Drivers:**
   - Switch to **🔍 Correlation & Drivers**. Explain the regression slope between **Attendance** and **Final Marks**, emphasizing why students with $<75\%$ attendance cluster in the lower-score region.
   - Review the Pearson correlation matrix heatmap.
4. **Academic Intervention (At-Risk Screening):**
   - Click **⚠️ Students Needing Improvement**. Show how the application automatically flags students needing help and generates concrete suggestions (e.g. *"Remedial tutoring needed in Maths"*).
   - Click the **Export At-Risk Student Report** button to demonstrate CSV export.
5. **Individual Student Lookup:**
   - Go to **👤 Individual Student Lookup**. Select a student (e.g., `STU1001`) to display their digital report card, class rank, percentile, and comparative bar chart vs. class averages.
6. **Real Dataset Ingestion & Cleaning:**
   - Switch the sidebar selector to **Option B: Upload Real Dataset**.
   - Click **Load Demo College CSV (`sample_upload.csv`)** to demonstrate the heuristic column auto-detection, missing value detection, duplicate removal, and median imputation engine in action.

---

## 🎓 College Viva Voce (Q&A) Reference

### Q1: What was the motivation behind this project?
> **Answer:** Educational institutions often struggle to identify struggling students early. This system provides a transparent, data-driven analytical tool that tracks student marks, analyzes the impact of study hours and attendance, and produces early intervention alerts before end-of-term examinations.

### Q2: Why did you choose Streamlit over Django or Flask?
> **Answer:** Streamlit allows rapid development of reactive, data-centric dashboards entirely in Python. Unlike Flask or Django, which require separate HTML/CSS/JavaScript frontends and REST APIs, Streamlit binds directly to Pandas DataFrames and Matplotlib figures, reducing boilerplate while delivering modern interactivity.

### Q3: How does your system handle custom real-world CSV files?
> **Answer:** The module `src/data_processor.py` implements a heuristic matching function `auto_detect_columns()` using fuzzy keyword searching. It automatically maps fields like `Roll_No` to `Student_ID` or `Math` to `Maths_Marks`. It also scans for missing values, duplicates, and out-of-range marks, providing interactive cleaning before updating the dashboard.

### Q4: How is the synthetic data generated?
> **Answer:** We used NumPy's `default_rng` with fixed seeds to ensure reproducibility. Rather than generating purely random numbers, we designed an underlying statistical relationship: attendance and study hours contribute to a base academic ability, to which subject-specific variance is added. This mirrors real-world classroom dynamics.

---

## 📄 License
Academic and Educational Demonstration Use.
Developed with Google Antigravity.
