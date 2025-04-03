import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb


# Load your data
df = pd.read_csv("../data/scores.csv")  # Assumes you have 'cv_text' and 'score' columns



# Generate embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Or any other model
X = embedder.encode(df["cv_text"].tolist(), show_progress_bar=True)

# Target labels (regression scores from 1–100)
y = df["score"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
model.save_model("cv_regression_model_2.json") 