from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def transform_data(df, target_column, scaler_save_path):
    # 1. Separate Features (X) and Target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # 2. Split the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Save the Scaler (Modular Path)
    # Ensure the models folder exists before saving
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    
    print(f"✅ Scaler saved at: {scaler_save_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test