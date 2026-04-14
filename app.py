from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# ---------------- MESSAGE FUNCTION ----------------
def funny_message(pred):
    if pred == "High Risk":
        return "⚠️ Oops... you're on HIGH RISK 😬 Your habits need attention. Try slowing down and taking care."
    elif pred == "Moderate Risk":
        return "🙂 Moderate risk. You're not in danger, but improving a few habits would help."
    else:
        return "😎 Low risk! You're doing great. Keep maintaining healthy habits."


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    # Get user input
    data = request.form.to_dict()

    # Convert to dataframe
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # ML prediction
    prediction = model.predict(df)[0]

    # Extract values
    smoking = data["Frequency"]
    alcohol = data["Consumption"]
    drugs = data["Use Frequency"]
    stress = data["Stress Level"]
    sleep = data["Sleep Hours"]
    physical = data["Physical Activity"]
    peer = data["Related to"]
    tobacco = data["Use of other Tobacco Products"]

    # ---------------- LOW RISK ----------------
    if (
        smoking == "0"
        and alcohol == "Never"
        and drugs == "Never"
        and stress == "Low"
        and peer == "No"
        and tobacco == "No"
        and physical in ["Moderate", "High"]
        and sleep in ["7-9", ">9"]
    ):
        prediction = "Low Risk"

    # ---------------- HIGH RISK ----------------
    elif (
        smoking in ["4", "5"]
        or alcohol == "Daily"
        or drugs == "Often"
        or (
            stress == "High"
            and sleep == "<5"
            and physical == "Low"
            and (smoking != "0" or alcohol != "Never" or drugs != "Never")
        )
    ):
        prediction = "High Risk"

    # ---------------- MODERATE ----------------
    else:
        prediction = "Moderate Risk"

    message = funny_message(prediction)

    return render_template(
        "index.html",
        result=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)