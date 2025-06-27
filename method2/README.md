
# 📦 Telco Customer Churn Prediction Dashboard

An interactive machine learning dashboard to predict customer churn for a telecommunications company. Built with **Flask + Tabulator.js**, supporting multiple models and real-time predictions.

---

## 📌 Features

- ✅ Interactive dashboard with filterable table
- ✅ Supports 4 models: Logistic Regression, SVM, Decision Tree, Random Forest
- ✅ Predict churn directly from the UI
- ✅ Clean UI with friendly prediction display
- ✅ Modular back-end with easy model swapping

---

## 🗂 Project Structure

```
.
├── app.py                           # Flask backend
├── Telco-Customer-Churn.csv        # Input dataset
├── train_all_telco_models.ipynb    # Model training notebook
├── models/
│   ├── logistic_model.pkl
│   ├── svm_model.pkl
│   ├── tree_model.pkl
│   └── rf_model.pkl
├── templates/
│   └── index.html                  # Frontend UI (Tabulator + JS)
```

---

## ⚙️ Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/shuyu-jpg/Group4_finalproject.git
cd Group4_finalproject/method2
```

2. **Install dependencies**

```bash
pip install flask pandas scikit-learn joblib
```

3. **Run the Flask app**

```bash
python app.py
```

Then open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📊 Model Details

All models are trained on `Telco-Customer-Churn.csv` using the following features:

- Customer Demographics (e.g., `SeniorCitizen`, `Partner`)
- Service Usage (e.g., `InternetService`, `StreamingTV`)
- Billing Info (e.g., `MonthlyCharges`, `Contract`)

The models support probability-based prediction via `.predict_proba()` and output:

- **Churn**
- **Stay**
- Confidence: shown as percentage in UI

---

## 💡 Example Prediction

```
📊 Prediction Result

Customer ID: 7590-VHVEG
Prediction: Churn
Confidence:
 - Stay: 35.8%
 - Churn: 64.2%
```

---

## 👨‍💻 Model Training

Run the included notebook:

```bash
jupyter notebook train_all_telco_models.ipynb
```

This notebook:
- Loads & cleans the dataset
- Trains 4 models
- Saves them as `.pkl` in the `models/` folder

---

## ✅ To-Do / Enhancements

- Add SweetAlert2 for better UI alerts
- Add charts to visualize churn breakdown
- Add model comparison in dashboard

---

## 📚 Credits

- Dataset: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- UI: Tabulator.js
- Backend: Flask, scikit-learn
