"""
Synthetic Dataset Generator for Student Performance Analysis
Generates realistic, reproducible student academic records.
"""

import numpy as np
import pandas as pd


FIRST_NAMES_MALE = [
    "Aarav", "Rohan", "Aditya", "Vikram", "Kabir", "Arjun", "Kunal", "Rahul",
    "Siddharth", "Dev", "Neil", "Aryan", "Varun", "Manish", "Amit", "Karan",
    "Pranav", "Nikhil", "Gaurav", "Harsh", "Sameer", "Rajesh", "Vivek", "Tanmay",
    "Aniket", "Deepak", "Ravi", "Suresh", "Yash", "Mohit"
]

FIRST_NAMES_FEMALE = [
    "Ananya", "Diya", "Isha", "Riya", "Pooja", "Sneha", "Tanvi", "Kavya",
    "Meera", "Shreya", "Aditi", "Neha", "Priya", "Nandini", "Simran", "Tara",
    "Ishita", "Anushka", "Divya", "Swati", "Rashmi", "Kriti", "Ritika", "Payal",
    "Bhavna", "Sunita", "Preeti", "Geeta", "Jyoti", "Pallavi"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Mehta", "Patel", "Singh", "Reddy", "Nair",
    "Iyer", "Chopra", "Joshi", "Bose", "Kulkarni", "Deshmukh", "Malhotra", "Saxena",
    "Kumar", "Pandey", "Mishra", "Das", "Rao", "Menon", "Bhat", "Kapoor",
    "Choudhury", "Bhattacharya", "Agarwal", "Pillai", "Shah", "Sen"
]


def calculate_grade(final_marks: float) -> str:
    """Assign letter grade based on final marks percentage."""
    if final_marks >= 90:
        return "A+"
    elif final_marks >= 80:
        return "A"
    elif final_marks >= 70:
        return "B"
    elif final_marks >= 60:
        return "C"
    elif final_marks >= 50:
        return "D"
    else:
        return "F"


def generate_synthetic_data(num_records: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student dataset with realistic relationships.
    
    Parameters:
        num_records: Number of student records (minimum 200).
        seed: Random seed for reproducible output.
        
    Returns:
        pd.DataFrame containing student academic data.
    """
    if num_records < 200:
        num_records = 200

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    records = []
    student_ids = [f"STU{1000 + i}" for i in range(1, num_records + 1)]
    genders = rng.choice(["Male", "Female"], size=num_records, p=[0.52, 0.48])

    for i in range(num_records):
        sid = student_ids[i]
        gender = genders[i]

        if gender == "Male":
            fname = rng.choice(FIRST_NAMES_MALE)
        else:
            fname = rng.choice(FIRST_NAMES_FEMALE)
        lname = rng.choice(LAST_NAMES)
        name = f"{fname} {lname}"

        # Study hours: normal distribution between 1 and 12 hours/week
        study_hours = float(np.clip(rng.normal(loc=6.2, scale=2.4), 1.0, 12.0))
        study_hours = round(study_hours, 1)

        # Attendance: beta-like distribution scaled to 45% - 100%
        # Most students have > 70% attendance, some fall below
        att_raw = rng.beta(a=7, b=2) * 55 + 45
        attendance = float(np.clip(att_raw, 45.0, 100.0))
        attendance = round(attendance, 1)

        # Base performance influenced by attendance and study hours
        # Attendance contributes ~35%, Study hours contributes ~45%, inherent variance ~20%
        base_ability = (attendance / 100.0) * 35.0 + (study_hours / 12.0) * 45.0 + rng.normal(0, 8.0)

        # Subject marks with subject-specific variations
        # Maths often has higher variance
        maths = int(np.clip(round(base_ability + rng.normal(0, 9.0) + 10), 25, 100))
        science = int(np.clip(round(base_ability + rng.normal(0, 7.5) + 12), 28, 100))
        english = int(np.clip(round(base_ability + rng.normal(2, 6.0) + 15), 30, 100))
        computer = int(np.clip(round(base_ability + rng.normal(1, 8.0) + 14), 26, 100))

        # Final marks is average of the four subjects
        final_marks = round((maths + science + english + computer) / 4.0, 1)
        grade = calculate_grade(final_marks)

        records.append({
            "Student_ID": sid,
            "Student_Name": name,
            "Gender": gender,
            "Attendance": attendance,
            "Study_Hours": study_hours,
            "Maths_Marks": maths,
            "Science_Marks": science,
            "English_Marks": english,
            "Computer_Marks": computer,
            "Final_Marks": final_marks,
            "Grade": grade
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df_sample = generate_synthetic_data(250, seed=42)
    print(f"Generated {len(df_sample)} records.")
    print(df_sample.head())
