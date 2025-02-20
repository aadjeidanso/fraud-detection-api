from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for frontend-backend communication
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow React requests

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print("Received data:", data)  # Debugging

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Ensure feature order matches training data
        expected_features = model.feature_names_in_
        df = df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of fraud

        print(f"Prediction: {prediction}, Probability: {probability}")  # Debugging

        # Return response
        return jsonify({
            "fraud_prediction": int(prediction),
            "fraud_probability": round(probability, 4)
        })
    
    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == '__main__':
    app.run(debug=True)
