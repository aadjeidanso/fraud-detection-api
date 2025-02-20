from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

# Mapping for categorical features
category_mappings = {
    "Low": -5,
    "Medium-Low": -2.5,
    "Medium": 0,
    "Medium-High": 2.5,
    "High": 5
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Convert categorical values to numbers
        for key in data:
            if data[key] in category_mappings:
                data[key] = category_mappings[data[key]]

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Ensure feature order matches training data
        expected_features = model.feature_names_in_
        df = df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of fraud

        return jsonify({
            "fraud_prediction": int(prediction),
            "fraud_probability": round(probability, 4)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
