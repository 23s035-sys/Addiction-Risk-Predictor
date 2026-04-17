from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# ---------------- LOAD MODEL FILES ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# ---------------- MESSAGE FUNCTION ----------------
def funny_message(pred):
    if pred == "High Risk":
        return "⚠️ High risk detected. Please take immediate steps to improve your lifestyle."
    elif pred == "Moderate Risk":
        return "🙂 Moderate risk. You're doing okay, but some improvements are needed."
    else:
        return "😎 Low risk! Keep maintaining your healthy habits."


# ---------------- RISK + REASON FUNCTION ----------------
def calculate_risk_and_reason(data):
    score = 10  # base risk
    reasons = []

    # Smoking
    if data["Frequency"] in ["4", "5"]:
        score += 25
        reasons.append("high smoking")
    else:
        reasons.append("controlled smoking habits")

    # Alcohol
    if data["Consumption"] == "Daily":
        score += 20
        reasons.append("daily alcohol use")
    else:
        reasons.append("limited alcohol consumption")

    # Drugs
    if data["Use Frequency"] == "Often":
        score += 25
        reasons.append("frequent drug use")
    else:
        reasons.append("no/low drug usage")

    # Sleep
    if data["Sleep Hours"] == "<5":
        score += 15
        reasons.append("poor sleeping habits")
    else:
        reasons.append("good sleeping pattern")

    # Physical Activity
    if data["Physical Activity"] == "Low":
        score += 10
        reasons.append("low physical activity")
    else:
        reasons.append("active lifestyle")

    # Stress
    if data["Stress Level"] == "High":
        score += 10
        reasons.append("high stress level")
    else:
        reasons.append("manageable stress level")

    score = min(score, 100)

    # Decide level
    if score < 30:
        level = "Low Risk"
    elif score < 70:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    return level, score, reasons


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.form.to_dict()

    # Convert to dataframe
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # ML prediction (optional, not used for final display)
    model.predict(df_scaled)

    # Custom logic
    level, score, reasons = calculate_risk_and_reason(data)

    message = funny_message(level)

    # Show only top 2 reasons
    reason_text = " and ".join(reasons[:2])

    return render_template(
        "index.html",
        result=level,
        message=message,
        score=score,
        reasons=reason_text
    )


if __name__ == "__main__":
    app.run(debug=True)
