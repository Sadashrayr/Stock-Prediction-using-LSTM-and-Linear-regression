from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load the models and scaler
linear_regression_model = joblib.load("linear_regression_model.pkl")
lstm_model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")  # Pre-fit scaler for consistent transformations


@app.route("/")
def home():
    return render_template("index.html")


import traceback  # Add this at the top of your app.py file


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("data", [])
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Ensure the input data is valid
        try:
            data = np.array(data).reshape(-1, 1)
        except ValueError:
            return (
                jsonify({"error": "Invalid data format. Provide numeric values."}),
                400,
            )

        # Scale the input data
        scaled_data = scaler.transform(data)

        # Prepare data for predictions
        if len(scaled_data) < 60:
            # Pad with zeros if less than 60 features
            padding = np.zeros((60 - len(scaled_data), 1))
            X_lr = np.vstack((padding, scaled_data)).reshape(1, -1)
        else:
            # Take only the last 60 features
            X_lr = scaled_data[-60:].reshape(1, -1)

        # Prepare data for LSTM
        X_lstm = scaled_data[-60:].reshape(1, -1, 1)

        # Predict using models
        lr_pred = linear_regression_model.predict(X_lr)
        lstm_pred = lstm_model.predict(X_lstm)

        # Inverse transform predictions
        lr_pred = scaler.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
        lstm_pred = scaler.inverse_transform(lstm_pred).flatten()

        return jsonify(
            {
                "linear_regression_prediction": lr_pred.tolist(),
                "lstm_prediction": lstm_pred.tolist(),
            }
        )
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
