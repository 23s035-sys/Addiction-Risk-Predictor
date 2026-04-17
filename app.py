from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


def funny_message(pred):
    if pred == "High Risk":
        return "⚠️ High risk detected. Take action immediately."
    elif pred == "Moderate Risk":
        return "🙂 Moderate risk. Improve a few habits."
    else:
        return "😎 Low risk! Keep it up."


# 🔥 CORE LOGIC WITH CONTRIBUTION %
def calculate_risk_and_reason(data):
    score = 10
    contributions = []

    # Smoking
    if data["Frequency"] in ["4", "5"]:
        score += 25
        contributions.append(("High Smoking", 25))
    elif data["Frequency"] in ["2", "3"]:
        score += 10
        contributions.append(("Moderate Smoking", 10))

    # Alcohol
    if data["Consumption"] == "Daily":
        score += 20
        contributions.append(("Daily Alcohol", 20))
    elif data["Consumption"] in ["Weekly", "Occasional"]:
        score += 8
        contributions.append(("Moderate Alcohol", 8))

    # Drugs
    if data["Use Frequency"] == "Often":
        score += 25
        contributions.append(("Frequent Drug Use", 25))
    elif data["Use Frequency"] == "Sometimes":
        score += 10
        contributions.append(("Moderate Drug Use", 10))

    # Sleep
    if data["Sleep Hours"] == "<5":
        score += 15
        contributions.append(("Poor Sleep", 15))
    elif data["Sleep Hours"] == "5-7":
        score += 8
        contributions.append(("Moderate Sleep", 8))

    # Physical Activity
    if data["Physical Activity"] == "Low":
        score += 10
        contributions.append(("Low Activity", 10))
    elif data["Physical Activity"] == "Moderate":
        score += 5
        contributions.append(("Moderate Activity", 5))

    # Stress
    if data["Stress Level"] == "High":
        score += 10
        contributions.append(("High Stress", 10))
    elif data["Stress Level"] == "Moderate":
        score += 5
        contributions.append(("Moderate Stress", 5))

    score = min(score, 100)

    if score < 30:
        level = "Low Risk"
    elif score < 70:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    return level, score, contributions


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df_scaled = scaler.transform(df)

    model.predict(df_scaled)

    level, score, contributions = calculate_risk_and_reason(data)

    message = funny_message(level)

    return render_template(
        "index.html",
        result=level,
        message=message,
        score=score,
        contributions=contributions
    )


if __name__ == "__main__":
    app.run(debug=True)
