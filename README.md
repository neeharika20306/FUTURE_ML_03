# AI Resume Screening & Candidate Ranking System

## Overview

This project is an NLP and Machine Learning based Resume Screening System that automatically evaluates resumes against a job description and ranks candidates based on their suitability for the role.

The system performs:

* Resume text cleaning and preprocessing
* Skill extraction
* Resume-job description matching
* Candidate ranking
* Skill gap identification
* Data visualization

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* NLP-based Text Processing

## Features

### Resume Parsing & Cleaning

Preprocesses resume text by removing unwanted characters and normalizing text.

### Skill Extraction

Extracts technical skills from resumes using a predefined skill database.

### Resume Matching

Uses TF-IDF Vectorization and Cosine Similarity to compare resumes with a job description.

### Candidate Ranking

Generates a final score based on:

* Skill Match Score
* Resume Similarity Score
* Category Relevance

### Skill Gap Analysis

Identifies missing skills required for the target role.

### Visualizations

* Top 10 Ranked Candidates
* Most Common Skills Distribution

## Dataset

This project uses the Resume Dataset available on Kaggle.

Dataset Source:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

The dataset contains 2484 resumes across multiple professional domains including:

* Information Technology
* Engineering
* Finance
* HR
* Banking
* Sales
* Healthcare
* Business Development
* And more

**Note:** The dataset file (`resume.csv`) is not included in this repository due to size and licensing considerations.

To run this project:

1. Download the dataset from Kaggle.
2. Create a folder named `Resume` in the project root directory.
3. Place `resume.csv` inside the folder.

Expected structure:

Resume_Screening_System/
│
├── Resume/
│   └── resume.csv
│
├── main.py
├── README.md
└── requirements.txt


## Sample Output

Top Candidate:

* Skill Match Percentage: 55.56%
* Matched Skills: Python, SQL, Pandas, Machine Learning, Data Analysis
* Missing Skills: Git, Deep Learning, Scikit-Learn, NumPy

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Outputs Generated

* ranked_candidates.csv
* top10_candidates.png
* skill_distribution.png
* top_candidate_analysis.txt
