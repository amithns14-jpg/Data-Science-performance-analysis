"""
Script to pre-generate synthetic and sample CSV files for testing and immediate demo.
Uses Python standard library (csv, random) so it runs with zero external dependencies.
"""

import csv
import random
import os

random.seed(42)

os.makedirs("data", exist_ok=True)

FIRST_NAMES_M = [
    "Aarav", "Rohan", "Aditya", "Vikram", "Kabir", "Arjun", "Kunal", "Rahul",
    "Siddharth", "Dev", "Neil", "Aryan", "Varun", "Manish", "Amit", "Karan",
    "Pranav", "Nikhil", "Gaurav", "Harsh", "Sameer", "Rajesh", "Vivek", "Tanmay"
]
FIRST_NAMES_F = [
    "Ananya", "Diya", "Isha", "Riya", "Pooja", "Sneha", "Tanvi", "Kavya",
    "Meera", "Shreya", "Aditi", "Neha", "Priya", "Nandini", "Simran", "Tara",
    "Ishita", "Anushka", "Divya", "Swati", "Rashmi", "Kriti", "Ritika", "Payal"
]
LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Mehta", "Patel", "Singh", "Reddy", "Nair",
    "Iyer", "Chopra", "Joshi", "Bose", "Kulkarni", "Deshmukh", "Malhotra", "Saxena",
    "Kumar", "Pandey", "Mishra", "Das", "Rao", "Menon", "Bhat", "Kapoor"
]

def calc_grade(m):
    if m >= 90: return "A+"
    elif m >= 80: return "A"
    elif m >= 70: return "B"
    elif m >= 60: return "C"
    elif m >= 50: return "D"
    else: return "F"

# 1. Generate synthetic_students.csv (250 records)
synthetic_path = os.path.join("data", "synthetic_students.csv")
with open(synthetic_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Student_ID", "Student_Name", "Gender", "Attendance", "Study_Hours",
        "Maths_Marks", "Science_Marks", "English_Marks", "Computer_Marks",
        "Final_Marks", "Grade"
    ])
    for i in range(1, 251):
        sid = f"STU{1000 + i}"
        gender = "Male" if random.random() < 0.52 else "Female"
        fname = random.choice(FIRST_NAMES_M if gender == "Male" else FIRST_NAMES_F)
        lname = random.choice(LAST_NAMES)
        name = f"{fname} {lname}"

        study_hours = round(max(1.0, min(12.0, random.gauss(6.5, 2.3))), 1)
        att_raw = random.betavariate(6.5, 2.0) * 55 + 45
        attendance = round(max(45.0, min(100.0, att_raw)), 1)

        base_ability = (attendance / 100.0) * 35.0 + (study_hours / 12.0) * 45.0 + random.gauss(0, 7.0)

        maths = int(max(25, min(100, round(base_ability + random.gauss(0, 9.0) + 10))))
        science = int(max(28, min(100, round(base_ability + random.gauss(0, 7.5) + 12))))
        english = int(max(30, min(100, round(base_ability + random.gauss(2, 6.0) + 15))))
        computer = int(max(26, min(100, round(base_ability + random.gauss(1, 8.0) + 14))))

        final_marks = round((maths + science + english + computer) / 4.0, 1)
        grade = calc_grade(final_marks)

        writer.writerow([
            sid, name, gender, attendance, study_hours,
            maths, science, english, computer, final_marks, grade
        ])

print(f"Created {synthetic_path} with 250 records.")

# 2. Generate sample_upload.csv (50 records with custom column names, a few missing values & duplicates to showcase cleaning)
upload_path = os.path.join("data", "sample_upload.csv")
with open(upload_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Roll_No", "Full_Name", "Sex", "Attendance_Percentage", "Weekly_Study_Hours",
        "Mathematics", "Physics_Chemistry", "English_Lang", "Computer_Science"
    ])
    
    rows = []
    for i in range(1, 49):
        roll = f"CS-2026-{100 + i}"
        gender = "M" if random.random() < 0.5 else "F"
        fname = random.choice(FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F)
        lname = random.choice(LAST_NAMES)
        name = f"{fname} {lname}"
        
        # introduce an occasional missing attendance or study hours
        att = "" if i == 7 else str(round(random.uniform(55, 98), 1))
        hrs = "" if i == 15 else str(round(random.uniform(2, 11), 1))
        
        m_score = random.randint(35, 98)
        s_score = random.randint(38, 99)
        e_score = random.randint(45, 95)
        c_score = random.randint(40, 100)

        rows.append([roll, name, gender, att, hrs, m_score, s_score, e_score, c_score])
    
    # Add 2 duplicate rows to test duplicate detection
    rows.append(rows[2])
    rows.append(rows[8])

    for r in rows:
        writer.writerow(r)

print(f"Created {upload_path} with {len(rows)} records (including simulated missing values & duplicates).")
