import pandas as pd
import joblib
from app.model import train_model

# Sample Data
data = {
    "hours": [1, 2, 3, 4, 5],
    "marks": [20, 40, 60, 80, 100]
}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["marks"]

# Train Model
model = train_model(X, y)

# Save Model
joblib.dump(model, "models/model.pkl")

print("Model Trained Successfully")