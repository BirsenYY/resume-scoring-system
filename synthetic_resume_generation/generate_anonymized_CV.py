import os
from openai import OpenAI
import random
from pathlib import Path
from natsort import natsorted

# Initialize OpenAI

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_DIR = "../data/test_resumes/"
OUTPUT_DIR = "../data/input_CVs_extra/"
NUM_SYNTHETIC_CVS = 20  # Number of synthetic CVs you want to generate

Path(OUTPUT_DIR).mkdir(exist_ok=True)
def load_txt_cvs(input_dir):
    cvs = []
    for filename in natsorted(os.listdir(input_dir)):
        print(filename)
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                cvs.append(f.read())
    return cvs

def generate_synthetic_cv(prompt_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic CVs."},
                {"role": "user", "content":  f"""Based on the following CV, generate a new and realistic CV with a similar structure and job domain.

However, make sure to:
- Add "NAME" instead of names and "EMAIL" instead of email addresses which means DO NOT ADD the NAME and SURNAME of the candidate. All synthetic CVs must have the same label for names and surnames. 
- Vary the number of certificates (include 0 to 5 randomly)
- Use different certificate names from the original CV
- Include a mix of popular and less common certifications
- Optionally invent plausible certificates if neededs
- Add side projects with variable 
- Vary the years of experience in a believable way
- Vary the past workplaces and side projects with variable numbers\n\n{prompt_text}"""}
            ],
            temperature=0.9,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return None

def save_cv(text, j, i):
    filename = f"synthetic_cv_{j}_{i}.txt"
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(text)

def run_pipeline():
    examples = load_txt_cvs(INPUT_DIR)
    for j in range(14, len(examples)):
        for i in range(1, 1 + NUM_SYNTHETIC_CVS):
            synthetic_cv = generate_synthetic_cv(examples[j])
            if synthetic_cv:
               save_cv(synthetic_cv, j + 1, i)
               print(f"Saved: synthetic_cv_{j + 1}_{i}.txt")

if __name__ == "__main__":
    run_pipeline()
