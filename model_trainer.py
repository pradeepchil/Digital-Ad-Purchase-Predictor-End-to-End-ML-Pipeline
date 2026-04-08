from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save_model(X_train, y_train, model_save_path):
    # 1. Initialize the Model
    model = LogisticRegression()
    
    # 2. Train (Fit) the Model
    model.fit(X_train, y_train)
    
    # 3. Ensure the models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 4. Save the Model
    joblib.dump(model, model_save_path)
    
    print(f"✅ Model trained and saved successfully at: {model_save_path}")
    return model