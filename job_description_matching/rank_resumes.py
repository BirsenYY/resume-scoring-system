from sentence_transformers import SentenceTransformer, util
import torch

# Load the SBERT model (fast + accurate)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example job description
job_description = """
We are looking for a Senior Software Engineer (Developer), Machine Learning Engineer, or Data Scientist with experience in NLP, deep learning, and real-world deployments.

The ideal candidate will:

Have worked in hands-on engineering roles building and deploying AI systems

Possess strong knowledge of Python, ML frameworks (PyTorch, TensorFlow), and cloud platforms

Demonstrate real-world impact through achievements such as Kaggle competitions, open-source contributions, or impactful personal projects

Hold a degree in Computer Science or a related STEM field from a reputable institution

Have experience at a top tech company or in high-performance teams

Note: This role is not focused on quality assurance or software test automation. Candidates with testing-centric backgrounds may not be aligned with the core responsibilities of this position.

We seek engineers who design and build intelligent systems — not just validate them.
"""

# Example list of CVs
cv_texts = [
    "Software developer with 3 years of Python experience. Built a chatbot using HuggingFace Transformers. Degree in Physics.",
    "Senior machine learning engineer with 10 years of experience. Led a team deploying NLP models in production. CS graduate from Stanford.",
    "Junior software engineer. Won a gold medal in a Kaggle NLP competition. Built ML pipelines with Scikit-learn and TensorFlow.",
    "Experienced QA professional with over 6 years of experience in software quality assurance and test automation. Adept in writing test plans, developing automated test scripts using Selenium and Java, and ensuring product stability through rigorous testing procedures.",
    "Mechanical Engineer with 8 years of experience in industrial design, thermodynamics, and HVAC system optimization. Skilled in CAD tools such as AutoCAD, SolidWorks, and ANSYS. Proven track record of managing large-scale mechanical installations and coordinating with manufacturing teams. Strong understanding of material science, fluid mechanics, and structural analysis. Passionate about sustainable energy solutions and mechanical system efficiency."
]

# Step 1: Encode the job description
jd_embedding = model.encode(job_description, convert_to_tensor=True)

# Step 2: Encode all CVs
cv_embeddings = model.encode(cv_texts, convert_to_tensor=True)

# Step 3: Compute cosine similarities
cosine_scores = util.cos_sim(jd_embedding, cv_embeddings)[0]

# Step 4: Rank CVs by similarity score
ranked_indices = torch.argsort(cosine_scores, descending=True)

print("\n🔍 Ranked CVs (most to least relevant):\n")
for rank, idx in enumerate(ranked_indices):
    print(f"Rank {rank+1} | Score: {cosine_scores[idx]:.3f}")
    print(f"CV: {cv_texts[idx][:120]}...\n")  # show preview of CV
