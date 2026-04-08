from fastapi import FastAPI
from app.schemas import AdPredictionInput, AdPredictionOutput
from app.engine import predict_purchase

app = FastAPI(title="Digital Ad Purchase Predictor")

@app.get("/")
def home():
    return {"message": "API is running! Use /predict for results."}

@app.post("/predict", response_model=AdPredictionOutput)
def predict(data: AdPredictionInput):
    # Call our engine
    pred, label, prob = predict_purchase(data.Age, data.Salary)
    
    return {
        "prediction": int(pred),
        "prediction_label": label,
        "probability": float(prob)
    }