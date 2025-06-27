# 📊 Telco Customer Churn Prediction Project

## 📌 Overview
A machine learning pipeline to predict customer churn for a telecommunications company. Completed by Group 4 using the CRISP-DM framework.

### 🔑 Key Features
- Data preprocessing & feature engineering
- Model training and comparison: Logistic Regression, Decision Tree, Random Forest, SVM
- Performance evaluation with metrics (Accuracy, F1-score, ROC-AUC)
- Model registration using `jrjModelRegistry`
- FastAPI demo interface with integrated dashboard

---

## 📂 Project Structure

```
├── Telco-Customer-Churn.csv       # Dataset
├── finalcoding2.ipynb             # Logistic Regression + Decision Tree
├── SVM.ipynb                      # Support Vector Machine model
├── train_all_models.py            # (Optional) Run all models together
├── main.py                        # FastAPI app with dashboard
└── .env-live                      # Environment config (Mongo/S3)
```

---

## 📊 Dataset Description

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records**: 7,043 customer entries
- **Features**: 21 original → 30 after encoding
- **Target variable**: `Churn` (Yes → 1, No → 0)

---

## ⚙️ Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

---

## 🧠 Evaluation Metrics

| Metric      | Description                                   |
|-------------|-----------------------------------------------|
| Accuracy    | Overall prediction correctness                |
| Precision   | True positive rate among predicted positives  |
| Recall      | True positive rate among actual positives     |
| F1-score    | Harmonic mean of precision and recall         |
| ROC-AUC     | Performance across all classification thresholds |

- ✅ **SVM** achieved best recall (0.79) and F1-score (0.62)
- ✅ **Logistic Regression** had highest ROC-AUC (0.83)

---

## 🧪 Model Registration

Models are registered using `jrjModelRegistry`. Example usage:

```python
from jrjModelRegistry import registerAJrjModel

registerAJrjModel(
    model_object,
    {
        "modelName": "YourName_ModelName_TelcoChurn",
        "version": "1.0.0",
        "metrics": {
            "accuracy": 0.79,
            "f1": 0.62
        }
    }
)
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or manually:
pip install pandas scikit-learn fastapi uvicorn python-dotenv jrjModelRegistry
```

### 2. Run Jupyter Notebooks

Open and execute:

- `finalcoding2.ipynb`
- `SVM.ipynb`

They will handle:
- Data preprocessing
- Model training
- Evaluation and registration

### 3. Start FastAPI Server

```bash
uvicorn main:app --reload

fast
```
or

fastapi dev main.py

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)  
Model dashboard is available via `handleDashboard(app)` integration.

---

## 🔁 Model Training Process Summary

### 🔹 Step 1: Preprocessing
- Drop `customerID`
- Convert `TotalCharges` to float and handle missing values
- One-hot encode categorical columns
- Standardize `tenure`, `MonthlyCharges`, `TotalCharges`
- Split into 80% train / 20% test with `random_state=42`

### 🔹 Step 2: Train Models
- Notebooks: `finalcoding2.ipynb`, `SVM.ipynb`
- Hyperparameter tuning (e.g., C, max_depth)
- Evaluate with multiple metrics
- Register models using `jrjModelRegistry`

### 🔹 Step 3: (Optional) Retrain via Script
```bash
python train_all_models.py
```


---

## 📎 License
MIT License