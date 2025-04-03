import requests
import time
from pathlib import Path
import csv


start_time = time.time()
# Set your job description
job_description = """
Role: Software Engineer / Machine Learning Engineer / Data Scientist  

We are seeking a high-performing Software Engineer, Machine Learning Engineer, or Data Scientist to join our AI-driven development team. This role focuses on the design, deployment, and continuous optimization of production-grade NLP systems.

Our candidate scoring system places heavy emphasis on job title relevance, demonstrable achievements, educational background, and real-world experience. While exceptional accomplishments can boost a candidate’s score, certain profiles are explicitly excluded.

What We're Looking For
Seniority & Title Relevance
Candidates with the following titles are prioritized:

CTO — highest priority

Principal Software Engineer / Principal ML or AI Engineer

Senior Software Engineer / Senior Machine Learning Engineer / Lead Data Scientist

Junior Software Engineer / Junior ML Engineer / Junior AI Engineer / Junior Data Scientist — accepted but require strong complementary achievements

Note: Software Developer is treated the same as Software Engineer.

Achievements & Demonstrable Impact
Candidates can score highly (even without senior titles) if they have:

Top placements in Kaggle competitions

Patents or peer-reviewed research publications

Substantial open-source contributions or GitHub portfolios

Academic recognition in AI/ML/NLP fields

Experience contributing to large-scale production ML/NLP systems

Educational Background
A Computer Science degree earns the highest score

Other STEM degrees (Engineering, Math, Physics, etc.) are also positively weighted

Graduating from a top-ranked university adds further value

Industry & Technical Experience
Highly scored candidates typically have:

Several years of relevant experience in software, ML, or data science roles

Time spent at top-tier tech companies or in elite engineering teams

Experience deploying machine learning systems at scale, especially in cloud environments like AWS, GCP, or Azure

Important Exclusions
Students are not accepted, regardless of Computer Science background or side projects

Candidates whose backgrounds are primarily in QA, testing, or manual testing are not considered, especially if titles include:

QA Engineer

Test Engineer

QA Analyst

Automation Tester

Titles emphasizing Selenium, test case design, etc.

However, candidates formerly in QA roles but who currently hold accepted job titles (e.g., Software Engineer, Data Scientist) may be scored positively, depending on:

Number of years in their current (valid) role

Computer Science or related STEM degree

Strength of personal projects, certifications, and real-world impact

"""


# Define the Ollama endpoint and model
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def build_prompt(jd, cv):
    return f"""
You are a CV screening assistant.

Job Description:
{jd}

Candidate CV:
{cv}

Question:

Rate how well this candidate matches the job description on a scale from 1 to 100. You can use **any number** between 1 and 100, including values like 42, 67, 88, 93, etc.

Be strict and precise. Do not round to the nearest 5 or 10. Use fine-grained numbers that reflect even small differences in relevance.

Return only the number. 
"""

def query_ollama(prompt, temperature=0.5):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# Run scoring
print("\n🔍 Ranking CVs using Mistral (prompt-based):\n")
results = []

# Set the folder path
folder_path = Path('../data/input_CVs')

# Loop through all .txt files
for txt_file in folder_path.glob('*.txt'):
    
    with open(txt_file, 'r') as file:
        cv_text = file.read()
        prompt = build_prompt(job_description, cv_text)
        print(f"Scoring CV {txt_file.name}...")
        score = query_ollama(prompt)
        
    try:
        score_float = float(score)
    except ValueError:
        score_float = 0.0  # fallback if model outputs text
    results.append((txt_file.name, cv_text, score_float))
    time.sleep(1)  # prevent overloading the model

# Sort by score
ranked = sorted(results, key=lambda x: x[2], reverse=True)

# Print results
for rank, (file_name, _, score) in enumerate(ranked):
    print(f"\nRank {rank+1} | Score: {score:.3f}")
    print(f"CV: {file_name}...\n")

# Write to CSV
with open('../data/scores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'cv_text', 'score'])  # Header
    writer.writerows(ranked)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.4f} seconds")