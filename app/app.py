import numpy as np
import torch
import torch.nn as nn
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

class WeightedDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, feature_weights):
        super(WeightedDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.feature_weights = feature_weights

    def forward(self, x):
        weighted_x = x * self.feature_weights
        out = self.fc1(weighted_x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def load_model():
    input_size = 8
    hidden_size = 128
    output_size = 4
    feature_weights = torch.tensor([0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05, 0.02], dtype=torch.float32)
    
    model = WeightedDNN(input_size, hidden_size, output_size, feature_weights)
    model.load_state_dict(torch.load("disease_prediction_model.pth"))
    model.eval()
    
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    
    return model, scaler, encoder

model, scaler, encoder = load_model()

DISEASE_MESSAGES = {
    "Arrhythmia": "Abnormal heart rhythm detected. Please consult a cardiologist.",
    "Heart Failure": "Signs suggest heart failure. Immediate medical attention recommended.",
    "Coronary Artery Disease": "Indicators of coronary artery disease present. Schedule a cardiac evaluation.",
    "Normal": "No significant cardiac issues detected. Maintain regular check-ups."
}

@app.route("/")
def home():
    return "Welcome to the Health Prediction API! Use the /predict endpoint to get predictions."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        age = float(data["age"])
        weight = float(data["weight"])
        height = float(data["height"]) / 100
        heart_rate = float(data["hr"])
        spo2 = float(data["spo2"])
        temp = float(data["temp"])
        qt_interval = float(data["qt_interval"])
        st_segment = float(data["st_segment"])
        
        input_data = np.array([[age, weight, height, heart_rate, spo2, temp, qt_interval, st_segment]])
        
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_disease = encoder.categories_[0][predicted_class_idx]
            
            message = DISEASE_MESSAGES.get(predicted_disease, "Please consult a healthcare professional.")
            
            disease_probs = {disease: float(probabilities[i] * 100) for i, disease in enumerate(encoder.categories_[0])}
        
        return jsonify({
            "prediction": predicted_disease,
            "probabilities": disease_probs,
            "message": message
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)