from pydantic import BaseModel

class AdPredictionInput(BaseModel):
    # These must match the column names your model expects
    Age: int
    Salary: float

class AdPredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    probability: float