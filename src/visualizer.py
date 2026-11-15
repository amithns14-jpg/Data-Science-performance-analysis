"""
Visualization Module for Student Performance Analysis
Generates modern, aesthetic Matplotlib and Seaborn figures.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent, modern aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.autolayout": True
})

PRIMARY_COLOR = "#2563EB"   # Modern Blue
SECONDARY_COLOR = "#10B981" # Emerald Green
ACCENT_COLOR = "#F59E0B"    # Amber
DANGER_COLOR = "#EF4444"    # Rose Red
PURPLE_COLOR = "#8B5CF6"    # Purple


def plot_grade_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plot grade distribution as a stylish bar chart with counts and percentages."""
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=100)
    
    order = ["A+", "A", "B", "C", "D", "F"]
    grade_counts = df["Grade"].value_counts().reindex(order, fill_value=0)
    total = len(df) if len(df) > 0 else 1

    palette = ["#10B981", "#3B82F6", "#06B6D4", "#F59E0B", "#F97316", "#EF4444"]
    bars = ax.bar(grade_counts.index, grade_counts.values, color=palette, edgecolor="none", width=0.6, alpha=0.9)

    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.annotate(f"{h}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_title("Student Grade Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Letter Grade", labelpad=8)
    ax.set_ylabel("Student Count", labelpad=8)
    ax.set_ylim(0, max(grade_counts.values, default=10) * 1.25)
    sns.despine(top=True, right=True)
    return fig


def plot_grade_donut(df: pd.DataFrame) -> plt.Figure:
    """Plot grade breakdown as a modern donut chart."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    order = ["A+", "A", "B", "C", "D", "F"]
    grade_counts = df["Grade"].value_counts().reindex(order, fill_value=0)
    # Filter non-zero
    non_zero = grade_counts[grade_counts > 0]
    palette = {
        "A+": "#10B981", "A": "#3B82F6", "B": "#06B6D4",
        "C": "#F59E0B", "D": "#F97316", "F": "#EF4444"
    }
    colors = [palette.get(g, "#6B7280") for g in non_zero.index]

    wedges, texts, autotexts = ax.pie(
        non_zero.values,
        labels=non_zero.index,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("black")

    ax.set_title("Grade Share Proportion", fontweight="bold", pad=10)
    return fig


def plot_subject_averages(df: pd.DataFrame) -> plt.Figure:
    """Compare average marks scored across academic subjects."""
    subj_cols = [c for c in ["Maths_Marks", "Science_Marks", "English_Marks", "Computer_Marks"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=100)

    if not subj_cols:
        ax.text(0.5, 0.5, "No subject columns available", ha="center", va="center")
        return fig

    averages = [df[col].mean() for col in subj_cols]
    labels = [col.replace("_Marks", "") for col in subj_cols]
    colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"]

    bars = ax.bar(labels, averages, color=colors[:len(labels)], width=0.55, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_title("Subject-Wise Average Performance", fontweight="bold", pad=12)
    ax.set_xlabel("Subject", labelpad=8)
    ax.set_ylabel("Average Marks (out of 100)", labelpad=8)
    ax.set_ylim(0, 110)
    ax.axhline(50, color="#EF4444", linestyle="--", linewidth=1.2, label="Passing Mark (50)")
    ax.legend(loc="upper right", frameon=True)
    sns.despine(top=True, right=True)
    return fig


def plot_subject_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Boxplot showing distribution, median, spread, and outliers across subjects."""
    subj_cols = [c for c in ["Maths_Marks", "Science_Marks", "English_Marks", "Computer_Marks"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)

    if not subj_cols:
        ax.text(0.5, 0.5, "No subject data available", ha="center", va="center")
        return fig

    clean_names = {col: col.replace("_Marks", "") for col in subj_cols}
    melted = df[subj_cols].rename(columns=clean_names).melt(var_name="Subject", value_name="Marks")

    palette = ["#93C5FD", "#86EFAC", "#C4B5FD", "#FDE68A"]
    sns.boxplot(x="Subject", y="Marks", hue="Subject", data=melted, ax=ax, palette=palette, width=0.5, boxprops=dict(alpha=0.85), legend=False)
    sns.stripplot(x="Subject", y="Marks", data=melted, ax=ax, color="#1F2937", size=3, alpha=0.25, jitter=0.2)

    ax.set_title("Subject Score Spread & Outliers", fontweight="bold", pad=12)
    ax.set_ylabel("Marks Obtained", labelpad=8)
    ax.set_ylim(0, 105)
    ax.axhline(50, color="#EF4444", linestyle="--", linewidth=1, label="Passing Mark (50)")
    ax.legend(loc="lower right", frameon=True)
    sns.despine(top=True, right=True)
    return fig


def plot_attendance_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of Attendance vs Final Marks with regression trendline."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=100)

    if "Attendance" not in df.columns or "Final_Marks" not in df.columns:
        ax.text(0.5, 0.5, "Attendance or Final Marks missing", ha="center", va="center")
        return fig

    # Categorize attendance risk (< 75%)
    low_att = df["Attendance"] < 75
    ax.scatter(df.loc[~low_att, "Attendance"], df.loc[~low_att, "Final_Marks"],
               c="#2563EB", alpha=0.65, edgecolors="none", s=45, label="Adequate Attendance (>=75%)")
    ax.scatter(df.loc[low_att, "Attendance"], df.loc[low_att, "Final_Marks"],
               c="#EF4444", alpha=0.85, edgecolors="none", s=55, marker="^", label="Low Attendance (<75%)")

    # Linear trendline
    valid = df[["Attendance", "Final_Marks"]].dropna()
    if len(valid) > 1:
        m, b = np.polyfit(valid["Attendance"], valid["Final_Marks"], 1)
        x_vals = np.linspace(valid["Attendance"].min(), valid["Attendance"].max(), 100)
        ax.plot(x_vals, m * x_vals + b, color="#059669", linewidth=2.2, linestyle="-", label=f"Trendline (slope: {m:.2f})")

    ax.axvline(75, color="#DC2626", linestyle=":", linewidth=1.2, label="75% Attendance Threshold")
    ax.axhline(50, color="#F59E0B", linestyle=":", linewidth=1.2, label="50% Passing Marks")

    ax.set_title("Impact of Attendance on Final Marks", fontweight="bold", pad=12)
    ax.set_xlabel("Attendance Percentage (%)", labelpad=8)
    ax.set_ylabel("Final Marks", labelpad=8)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    sns.despine(top=True, right=True)
    return fig


def plot_study_hours_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of Study Hours vs Final Marks with trendline."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=100)

    if "Study_Hours" not in df.columns or "Final_Marks" not in df.columns:
        ax.text(0.5, 0.5, "Study Hours or Final Marks missing", ha="center", va="center")
        return fig

    sns.regplot(
        x="Study_Hours",
        y="Final_Marks",
        data=df,
        ax=ax,
        scatter_kws={"color": "#8B5CF6", "alpha": 0.6, "s": 45},
        line_kws={"color": "#D97706", "linewidth": 2.2}
    )

    ax.set_title("Study Hours vs Academic Outcome", fontweight="bold", pad=12)
    ax.set_xlabel("Weekly Study Hours (hrs)", labelpad=8)
    ax.set_ylabel("Final Marks", labelpad=8)
    ax.set_ylim(0, 105)
    sns.despine(top=True, right=True)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation matrix heatmap among all numerical performance parameters."""
    num_cols = ["Attendance", "Study_Hours", "Maths_Marks", "Science_Marks",
                "English_Marks", "Computer_Marks", "Final_Marks"]
    available = [c for c in num_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=100)
    if len(available) < 2:
        ax.text(0.5, 0.5, "Insufficient numerical columns for correlation", ha="center", va="center")
        return fig

    corr = df[available].corr()
    clean_labels = [c.replace("_Marks", "").replace("_", " ") for c in available]

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=0.75,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=ax
    )
    ax.set_title("Performance Metrics Correlation Matrix", fontweight="bold", pad=12)
    return fig


def plot_gender_comparison(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Compare performance across gender groups."""
    if "Gender" not in df.columns or "Final_Marks" not in df.columns:
        return None

    subj_cols = [c for c in ["Maths_Marks", "Science_Marks", "English_Marks", "Computer_Marks", "Final_Marks"] if c in df.columns]
    if not subj_cols:
        return None

    grouped = df.groupby("Gender")[subj_cols].mean()
    clean_names = [c.replace("_Marks", "").replace("_", " ") for c in subj_cols]
    grouped.columns = clean_names

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    grouped.T.plot(kind="bar", ax=ax, colormap="viridis", width=0.7, alpha=0.88)

    ax.set_title("Performance Comparison by Gender", fontweight="bold", pad=12)
    ax.set_ylabel("Average Marks", labelpad=8)
    ax.set_xlabel("Subject / Assessment", labelpad=8)
    ax.set_ylim(0, 105)
    ax.legend(title="Gender", frameon=True)
    plt.xticks(rotation=0)
    sns.despine(top=True, right=True)
    return fig


def plot_student_comparison(student_row: pd.Series, class_means: pd.Series) -> plt.Figure:
    """Compare a single student's subject scores against the class average."""
    subjects = ["Maths", "Science", "English", "Computer"]
    student_scores = []
    class_scores = []

    for s in subjects:
        col = f"{s}_Marks"
        student_scores.append(float(student_row.get(col, 0)))
        class_scores.append(float(class_means.get(col, 0)))

    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    bars1 = ax.bar(x - width/2, student_scores, width, label="Student Score", color="#3B82F6", alpha=0.9)
    bars2 = ax.bar(x + width/2, class_scores, width, label="Class Average", color="#9CA3AF", alpha=0.9)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)

    name = student_row.get("Student_Name", "Student")
    ax.set_title(f"Score Comparison: {name} vs Class Average", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("Marks (out of 100)", labelpad=8)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", frameon=True)
    sns.despine(top=True, right=True)
    return fig
