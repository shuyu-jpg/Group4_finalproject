from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 加载模型
log_model = joblib.load("logistic_model.pkl")
tree_model = joblib.load("tree_model.pkl")
svm_model = joblib.load("svm_model.pkl")

# 模型使用的特征
FEATURES = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

@app.route("/")
def home():
    return "✅ Churn Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.json
        model_type = input_json.get("model", "logistic")
        data = input_json["data"]

        # 强制转换为 DataFrame
        df = pd.DataFrame([data])
        df = df[FEATURES]

        # 模型选择
        if model_type == "logistic":
            pred = log_model.predict(df)
        elif model_type == "tree":
            pred = tree_model.predict(df)
        elif model_type == "svm":
            pred = svm_model.predict(df)
        else:
            return jsonify({"error": "Unknown model type"}), 400

        return jsonify({"prediction": pred.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
