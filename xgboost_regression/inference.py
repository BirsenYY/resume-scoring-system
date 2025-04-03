import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from pathlib import Path
import time

start_time = time.time()

def extract_text_file():
    cvs = []
    folder_path = Path('../data/test_resumes')
    for txt_file in folder_path.glob('*.txt'):
    
         with open(txt_file, 'r') as file:
            
            cvs.append([file.name, file.read()])
    return cvs


embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Or any other model
cvs = extract_text_file()
cv_texts = [text for _, text in cvs]

X = embedder.encode(cv_texts, show_progress_bar=True)

loaded_model = xgb.XGBRegressor()
loaded_model.load_model("cv_regression_model_2.json")
Y = loaded_model.predict(X)

with open('results_3.txt', 'w', encoding='utf-8') as f:
    for (filename, _), score in sorted(zip(cvs, Y), key=lambda x: x[1], reverse=True):
        f.write(f"{filename}: {score}\n")
        print(filename + ":  " + str(score))

end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")