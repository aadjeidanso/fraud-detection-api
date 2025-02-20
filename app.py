from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # Enable CORS to allow frontend requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained fraud detection model
model = joblib.load("fraud_detection_model.pkl")

# Define the feature order (ensure it matches the trained model)
expected_features = ["V1", "V2", "V3", "V4", "V10", "V12", "V14", "V17", "Amount"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert user-friendly names to match the model's expected feature names
        feature_mapping = {
            "transaction_smoothness": "V1",
            "spending_behavior_anomaly": "V2",
            "transaction_complexity_score": "V3",
            "unusual_transaction_score": "V4",
            "rapid_large_spending": "V10",
            "frequent_large_withdrawals": "V12",
            "high_risk_transaction_indicator": "V14",
            "authentication_irregularities": "V17",
            "transaction_amount": "Amount",
        }
        
        # Transform user input to match the model's expected format
        transformed_data = {feature_mapping[k]: v for k, v in data.items() if k in feature_mapping}
        
        # Convert to DataFrame and reindex to ensure feature order
        df = pd.DataFrame([transformed_data])
        df = df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]  # 0 = Legit, 1 = Fraud
        probability = model.predict_proba(df)[0][1]  # Probability of fraud

        # Return response with fraud probability
        return jsonify({
            "fraud_prediction": int(prediction),
            "fraud_probability": round(probability, 4)  # Round to 4 decimal places
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the API
if __name__ == '__main__':
    app.run(debug=True)
