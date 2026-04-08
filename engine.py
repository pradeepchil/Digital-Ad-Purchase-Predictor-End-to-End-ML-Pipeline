import joblib
import pandas as pd
from pathlib import Path

# Setup paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

def predict_purchase(age: int, salary: float):
    # 1. Load the artifacts
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Prepare the input
    input_data = pd.DataFrame([[age, salary]], columns=['Age', 'Salary'])
    
    # 3. Scale and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    # Get probability (optional but professional)
    prob = model.predict_proba(scaled_data)[0][1] 
    
    label = "Will Buy" if prediction == 1 else "Will Not Buy"
    
    return prediction, label, prob