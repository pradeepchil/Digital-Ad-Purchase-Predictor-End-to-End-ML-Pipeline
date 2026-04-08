# 1. Import your modular machines
from src.data_ingestion import ingest_data
from src.data_transformation import transform_data
from src.model_trainer import train_and_save_model

def run_training_pipeline():
    print("--- Starting ML Pipeline ---")

    # --- CONFIGURATION (Your Windows Paths) ---
    # Use 'r' before the string to avoid backslash errors
    RAW_DATA_PATH = r'C:\Users\prade\OneDrive\Desktop\sample_project\raw_data\DigitalAd_dataset.csv'
    CLEANED_DATA_PATH = r'C:\Users\prade\OneDrive\Desktop\sample_project\cleaned_data\cleaned_ads.csv'
    SCALER_PATH = r'C:\Users\prade\OneDrive\Desktop\sample_project\models\scaler.pkl'
    MODEL_PATH = r'C:\Users\prade\OneDrive\Desktop\sample_project\models\logistic_model.pkl'

    # --- STEP 1: INGESTION ---
    # This cleans the duplicates and saves the CSV
    df = ingest_data(RAW_DATA_PATH, CLEANED_DATA_PATH)

    # --- STEP 2: TRANSFORMATION ---
    # This splits the data and saves the scaler.pkl
    X_train, X_test, y_train, y_test = transform_data(
        df=df, 
        target_column='Status', 
        scaler_save_path=SCALER_PATH
    )

    # --- STEP 3: TRAINING ---
    # This fits the model and saves the logistic_model.pkl
    train_and_save_model(
        X_train=X_train, 
        y_train=y_train, 
        model_save_path=MODEL_PATH
    )

    print("--- Pipeline Execution Finished! ---")

if __name__ == "__main__":
    run_training_pipeline()