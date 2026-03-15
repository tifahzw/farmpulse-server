from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
with open("farmpulse_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ FarmPulse AI model loaded.")

@app.route("/")
def home():
    return "FarmPulse AI Server is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features in same order as training
        features = [[
            float(data["soilMoisture"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["pumpCurrent"]),
            int(data["hour"])
        ]]

        prediction   = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence   = round(max(probabilities) * 100, 1)

        # Generate recommendation
        pump_running = float(data["pumpCurrent"]) > 1.5

        if prediction == "HIGH":
            rec = "START irrigation NOW!" if not pump_running else "Irrigation ACTIVE. Continue."
        elif prediction == "MEDIUM":
            rec = "Irrigate within 2 hours." if not pump_running else "Monitor closely."
        elif prediction == "LOW":
            rec = "Monitor soil moisture."
        else:
            rec = "Soil adequate. No irrigation needed."

        return jsonify({
            "riskLevel":      prediction,
            "confidence":     confidence,
            "recommendation": rec,
            "status":         "ok"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)