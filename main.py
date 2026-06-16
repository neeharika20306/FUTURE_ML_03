import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("Resume/resume.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nCategories:")
print(df["Category"].value_counts())

# ==========================
# JOB DESCRIPTION
# ==========================

job_description = """
Machine Learning Intern

Required Skills:
Python
Machine Learning
Deep Learning
Data Analysis
Pandas
NumPy
Scikit-learn
SQL
Git
"""

# ==========================
# TEXT CLEANING
# ==========================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

df["clean_resume"] = df["Resume_str"].apply(clean_text)

# ==========================
# FILTER IT + ENGINEERING
# ==========================

it_df = df[
    (df["Category"] == "INFORMATION-TECHNOLOGY") |
    (df["Category"] == "ENGINEERING")
].copy()

print("\nIT + Engineering Resumes:", it_df.shape)

# ==========================
# SKILL DATABASE
# ==========================

skills_db = [
    "python",
    "java",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "machine learning",
    "deep learning",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "power bi",
    "excel",
    "git",
    "github",
    "data analysis",
    "nlp",
    "artificial intelligence",
    "computer vision",
    "flask",
    "django",
    "html",
    "css",
    "javascript"
]

# ==========================
# SKILL EXTRACTION
# ==========================

def extract_skills(text):

    found_skills = []

    text = text.lower()

    for skill in skills_db:

        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills

required_skills = extract_skills(job_description)

print("\nRequired Skills:")
print(required_skills)

# ==========================
# TF-IDF SCORING
# ==========================

scores = []

for resume in it_df["clean_resume"]:

    docs = [job_description, resume]

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(docs)

    score = cosine_similarity(
        matrix[0],
        matrix[1]
    )[0][0]

    scores.append(score)

it_df["Score"] = scores

# ==========================
# SKILL MATCHING
# ==========================

it_df["Skills"] = it_df["clean_resume"].apply(
    extract_skills
)

def calculate_skill_score(candidate_skills):

    matched = len(
        set(candidate_skills) &
        set(required_skills)
    )

    total = len(required_skills)

    return (matched / total) * 100

it_df["Skill_Score"] = it_df["Skills"].apply(
    calculate_skill_score
)

# ==========================
# FINAL SCORE
# ==========================

it_df["Skill_Score"] = it_df["Skills"].apply(
    calculate_skill_score
)

# Give bonus to IT resumes

it_df["Category_Bonus"] = 0

it_df.loc[
    it_df["Category"] == "INFORMATION-TECHNOLOGY",
    "Category_Bonus"
] = 10

# FINAL SCORE

it_df["Final_Score"] = (
    0.6 * it_df["Skill_Score"]
    +
    0.3 * (it_df["Score"] * 100)
    +
    0.1 * it_df["Category_Bonus"]
)

# ==========================
# RANK CANDIDATES
# ==========================
it_df["Category_Bonus"] = 0

it_df.loc[
    it_df["Category"] == "INFORMATION-TECHNOLOGY",
    "Category_Bonus"
] = 10

ranked = it_df.sort_values(
    by="Final_Score",
    ascending=False
)

print("\nTOP 10 CANDIDATES")
print(
    ranked[
        [
            "ID",
            "Category",
            "Skill_Score",
            "Final_Score"
        ]
    ].head(10)
)

# ==========================
# SKILL GAP ANALYSIS
# ==========================

top_candidate = ranked.iloc[0]

candidate_skills = top_candidate["Skills"]

missing_skills = list(
    set(required_skills)
    - set(candidate_skills)
)

matched_skills = list(
    set(required_skills)
    &
    set(candidate_skills)
)

print("\n==========================")
print("TOP CANDIDATE ANALYSIS")
print("==========================")

print("Candidate ID:",
      top_candidate["ID"])

print("\nMatched Skills:")
print(matched_skills)

print("\nMissing Skills:")
print(missing_skills)

match_percentage = (
    len(matched_skills)
    /
    len(required_skills)
) * 100

print(
    "\nSkill Match Percentage:",
    round(match_percentage, 2),
    "%"
)

with open("top_candidate_analysis.txt", "w") as f:

    f.write("TOP CANDIDATE ANALYSIS\n")
    f.write("=====================\n\n")

    f.write(f"Candidate ID: {top_candidate['ID']}\n\n")

    f.write("Matched Skills:\n")
    for skill in matched_skills:
        f.write(f"- {skill}\n")

    f.write("\nMissing Skills:\n")
    for skill in missing_skills:
        f.write(f"- {skill}\n")

    f.write(
        f"\nSkill Match Percentage: {round(match_percentage, 2)}%"
    )

print("Top Candidate Analysis saved.")

# ==========================
# TOP 10 GRAPH
# ==========================

top10 = ranked.head(10)

plt.figure(figsize=(10, 5))

plt.bar(
    top10["ID"].astype(str),
    top10["Final_Score"]
)

plt.xticks(rotation=45)

plt.title(
    "Top 10 Ranked Candidates"
)

plt.xlabel(
    "Candidate ID"
)

plt.ylabel(
    "Final Score"
)

plt.tight_layout()

plt.savefig(
    "top10_candidates.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Top 10 Candidates graph saved.")

from collections import Counter

all_skills = []

for skills in it_df["Skills"]:
    all_skills.extend(skills)

skill_counts = Counter(all_skills)

top_skills = skill_counts.most_common(10)

skills = [item[0] for item in top_skills]
counts = [item[1] for item in top_skills]

plt.figure(figsize=(10,5))
plt.bar(skills, counts)

plt.title("Top 10 Most Common Skills")
plt.xlabel("Skills")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "skill_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Skill Distribution graph saved.")

# ==========================
# SAVE RESULTS
# ==========================

ranked.to_csv(
    "ranked_candidates.csv",
    index=False
)

print(
    "\nResults saved as ranked_candidates.csv"
)