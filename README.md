Digital Ad Purchase Predictor: End-to-End ML Pipeline
📌 Project Overview
This project is a production-ready Machine Learning system designed to predict whether a user will purchase a product based on their Age and Salary. While the dataset is straightforward, the architecture follows professional MLOps standards, moving away from flat scripts into a modular, scalable system.

🏗️ Modular Architecture
To keep the project clean and maintainable, I have segmented the logic into distinct "machines":

src/data_ingestion.py: Handles raw data loading and initial cleaning.

src/data_transformation.py: Manages feature scaling and preprocessing to ensure data is "math-ready."

src/model_trainer.py: Trains the Logistic Regression model and saves the "Brain" (model) and "Memory" (scaler) as .pkl files.

app/: A dedicated deployment layer using FastAPI and Pydantic for real-time inference.

🛠️ Key Technical Features
Data Validation: Uses Pydantic schemas to enforce strict data types on API inputs, preventing server crashes from "dirty" data.

Path Portability: Utilizes Python's pathlib to dynamically resolve directory paths, making the project run seamlessly on Windows, Mac, or Linux.

Robust File Handling: Implemented os module checks to automatically generate necessary directories (like /models) if they are missing.

Serialized Inference: Separates the training phase from the prediction phase by using Joblib for model persistence.

🚀 How to Run
1. Setup Environment
Bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
2. Run the Training Pipeline
This will clean the data and generate the .pkl artifacts in the models/ folder.

Bash
python main_train.py
3. Start the API Server
Launch the FastAPI server to start making real-time predictions.

Bash
uvicorn app.main:app --reload
View the automatic documentation at: http://127.0.0.1:8000/docs

📊 Sample API Response
JSON
{
    "prediction": 1,
    "prediction_label": "Will Buy",
    "probability": 0.7376
}
👨‍💻 Author
Pradeep (Pradee)
Aspiring Data Professional | ML Engineering Student

