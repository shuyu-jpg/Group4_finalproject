
# Telco Customer Churn Prediction Project

## 📌 Overview
This project is a machine learning pipeline developed to predict customer churn for a telecommunications company. It was completed as part of a group final assignment using the CRISP-DM framework.

Key features include:
- Data preprocessing
- Feature engineering
- Multiple model training and evaluation
- Automatic model registration to a MongoDB model registry using `jrjModelRegistry`

## 📂 Project Structure

```
├── Telco-Customer-Churn.csv       # Dataset
├── Finaltask_coding.ipynb         # Jupyter Notebook with full ML pipeline
├── train_models_local.py          # Local model training script
├── train_all_models.py            # Train all models for comparison
├── register_models.py             # Register models manually
├── register_models_fixed.py       # Register models with fixes
├── logistic_model.pkl             # Trained Logistic Regression model
├── tree_model.pkl                 # Trained Decision Tree model
├── svm_model.pkl                  # Trained SVM model
├── main.py                        # FastAPI App (optional front-end)
├── templates/
│   └── form.html                  # HTML form for user input (if API is used)
└── ATT01786.env-live              # Environment variable config file (Mongo URI, S3 configs etc.)
```

## 📊 Dataset
- **Source**: Telco Customer Churn dataset
- Contains customer demographic, account, and service usage data.
- Target variable: `Churn`

## ⚙️ Models Used
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)

## 🧠 Evaluation
Each model was evaluated using:
- Accuracy
- F1-score
- Confusion Matrix
- Cross-Validation (5-fold)

Best model was registered with:
```json
{
  "modelName": "BestModel_CrossValidation",
  "version": "1.0.0",
  "score": -1 * MSE,
  "modelLibrary": "sklearn",
  "libraryMetadata": {
    "r_squared": ...
  }
}
```

## 🧪 Model Registry
Models are automatically registered to a remote MongoDB-based model registry:
```python
from jrjModelRegistry.jrjModelRegistry import registerAJrjModel

registerAJrjModel(
    model_object,
    metadata_dict
)
```

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or manually:
pip install pandas scikit-learn fastapi python-dotenv jrjModelRegistry
```

### 2. Load Notebook
Open and run `Finaltask_coding.ipynb` to train, evaluate, and register the best model.

### 3. Optional: Run API
If using FastAPI interface:
```bash
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/form` to use form.html.

## 👥 Contributors
Group 4 – Master in Digital Marketing & Data Science  
EMLYON Business School, 2025

## 📎 License
MIT License
