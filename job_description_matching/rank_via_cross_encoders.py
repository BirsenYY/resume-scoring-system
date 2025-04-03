from sentence_transformers import CrossEncoder

# Load pretrained cross-encoder
model = CrossEncoder("cross-encoder/stsb-roberta-base")  # or try other similar models

# Your job description
job_description = """
We are seeking a highly capable software engineer (developer), machine learning engineer, or data scientist to join our advanced AI team. The ideal candidate will bring a strong background in software development and machine learning, with a proven ability to deliver impactful results.

💡 What We’re Looking For:
Seniority matters: We value candidates who have held titles such as Senior Software Engineer, Senior Machine Learning Engineer, or Senior Data Scientist, as these often indicate depth of experience and responsibility. CTO-level leadership or technical strategy roles are also highly regarded.

Education: A degree in Computer Science is strongly preferred, but we also welcome applicants from other STEM backgrounds. Graduates from top-ranking universities will be prioritized.

Professional experience: Candidates who have worked at top-tier tech companies are especially encouraged to apply, as this often reflects a strong engineering foundation and the ability to operate at scale.

Achievements over titles: Exceptional accomplishments such as winning Kaggle competitions, building widely-used open-source tools, or publishing impactful ML research will carry significant weight in our evaluation.

Side projects: We value candidates who demonstrate passion for technology through relevant personal projects, hackathon participation, or freelance/consulting experience in relevant areas. 
Note: Quality assurance or software test engineers are not suitable for this role. 
"""

# List of CVs
cv_texts = [
    "Senior ML engineer with 10 years experience. Deployed transformer models in production. CS grad from Stanford.",
    "Junior dev. Kaggle gold medalist. Built NLP pipelines with PyTorch.",
    "QA engineer with 6 years of test automation experience in Selenium and Jenkins.",
    "Software developer with 3 years of Python and chatbot experience using HuggingFace.",
    "Mechanical Engineer with background in thermodynamics, SolidWorks, and HVAC systems."
]

# Format input pairs: (JD, CV)
input_pairs = [(job_description, cv) for cv in cv_texts]

# Predict similarity scores
scores = model.predict(input_pairs)

# Rank CVs by score
ranked = sorted(zip(cv_texts, scores), key=lambda x: x[1], reverse=True)

# Display ranked results
print("\n🔍 Ranked CVs (Cross-Encoder):\n")
for i, (cv, score) in enumerate(ranked):
    print(f"Rank {i+1} | Score: {score:.3f}")
    print(f"CV: {cv[:120]}...\n")
