from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# Message
def funny_message(pred):
    if pred == "High Risk":
        return "⚠️ High risk detected. Please improve your lifestyle."
    elif pred == "Moderate Risk":
        return "🙂 Moderate risk. Improve some habits."
    else:
        return "😎 Low risk! Keep it up."


# Risk calculation
def calculate_risk_and_reason(data):
    score = 0
    reasons = []

    if data["Frequency"] in ["4", "5"]:
        score += 25
        reasons.append("high smoking")

    if data["Consumption"] == "Daily":
        score += 20
        reasons.append("daily alcohol")

    if data["Use Frequency"] == "Often":
        score += 25
        reasons.append("drug usage")

    if data["Sleep Hours"] == "<5":
        score += 15
        reasons.append("poor sleep")

    if data["Physical Activity"] == "Low":
        score += 10
        reasons.append("low activity")

    if data["Stress Level"] == "High":
        score += 10
        reasons.append("high stress")

    score = min(score, 100)

    if score < 30:
        level = "Low Risk"
    elif score < 70:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    return level, score, reasons


# Home
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    data = request.form.to_dict()

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    df_scaled = scaler.transform(df)

    model_prediction = model.predict(df_scaled)[0]

    level, score, reasons = calculate_risk_and_reason(data)

    message = funny_message(level)

    reason_text = ", ".join(reasons) if reasons else "healthy lifestyle"

    return render_template(
        "index.html",
        result=level,
        message=message,
        score=score,
        reasons=reason_text
    )


if __name__ == "__main__":
    app.run(debug=True)
