import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Model file mapping
MODEL_MAP = {
    "logistic": "logistic_model.pkl",
    "svm": "svm_model.pkl",
    "tree": "tree_model.pkl",
    "rf": "rf_model.pkl"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    return jsonify(df.to_dict(orient="records"))

@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):
    try:
        row = request.json
        if model_type not in MODEL_MAP:
            return jsonify({"error": "Invalid model type"}), 400

        model_path = os.path.join("models", MODEL_MAP[model_type])
        model = joblib.load(model_path)

        input_df = pd.DataFrame([row])
        input_df = input_df.drop(columns=["customerID", "Churn"], errors="ignore")

        y_pred = model.predict(input_df)
        y_proba = model.predict_proba(input_df)[0].tolist() if hasattr(model, "predict_proba") else None

        return jsonify({
            "customerID": row.get("customerID", "unknown"),
            "prediction": int(y_pred[0]),
            "probability": y_proba
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
