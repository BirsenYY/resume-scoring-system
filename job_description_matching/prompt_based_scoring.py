import requests
import time
from pathlib import Path
import csv


start_time = time.time()
# Set your job description
job_description = """
We are seeking a high-performing Software Engineer, Machine Learning Engineer, or Data Scientist to join our AI-driven development team. This role involves the design, deployment, and improvement of production-level NLP systems. While job titles are important, we also consider exceptional achievements that demonstrate real-world impact.

What We're Looking For:
Seniority & Title Relevance
Candidates with senior-level titles are highly preferred:

CTO — highest priority

Principal Software Engineer  / Principal ML or AI Engineer

Senior Software Engineer / Senior Machine Learning Engineer  / Lead Data Scientist

We also accept junior software, ML, AI engineers, or data scientists, but they are not scored as highly unless paired with exceptional achievements.

Achievements & Demonstrable Impact:
Exceptional success may outweigh job titles in some cases. We give strong preference to candidates who have demonstrated clear and significant impact through:

Winning or ranking highly in Kaggle competitions

Having patents or published research papers

Contributing to open-source projects or maintaining significant GitHub portfolios

Being an academic in a relevant AI/ML/NLP field

Contributing to large-scale ML/NLP systems

Candidates without a relevant job title can still be prioritized if they show exceptional success in the areas above.

Educational Background:
A Computer Science degree carries the highest weight.

A degree in other STEM fields (Engineering, Math, Physics, etc.) is also positive, though not as strong as a Computer Science background.

Graduating from a top-ranked university further increases the score.

Relevant Industry Experience
Candidates with longer and relevant experience are scored higher.

Experience at top-tier tech companies or within high-performance teams is valued.

Experience deploying ML systems in production (e.g., at scale or in cloud environments like AWS, GCP, Azure) adds weight.

Important note: ML/AI/Software engineers are expressed as "developer" too. Therefore, let's say "sofware engineer" and "software developer" are same. 
Not a Fit:
Candidates whose backgrounds are primarily in QA, software testing, or manual testing are not considered.

Job titles like QA Engineer, Test Engineer, or those focused on Selenium, JUnit, or test case design are not aligned with this role.

Candidates with non-AI/ML/Software/Data Scientist titles are generally not accepted unless they have exceptional achievements (e.g., Kaggle winner, research pioneer).

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
for rank, (file_name,cv_text, score) in enumerate(ranked):
    print(f"\nRank {rank+1} | Score: {score:.3f}")
    print(f"CV: {cv_text[:120]}...\n")

# Write to CSV
with open('../data/scores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'cv_text', 'score'])  # Header
    writer.writerows(ranked)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.4f} seconds")