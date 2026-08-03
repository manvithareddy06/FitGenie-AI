from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("Calories_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Getting values from HTML form
    gender = float(request.form["gender"])
    age = float(request.form["age"])
    height = float(request.form["height"])
    weight = float(request.form["weight"])
    duration = float(request.form["duration"])
    heartrate = float(request.form["heart_rate"])
    bodytemperature = float(request.form["body_temp"])

    # Same order as model training
    input_data = [[
        gender,
        age,
        height,
        weight,
        duration,
        heartrate,
        bodytemperature
    ]]

    # Prediction
    prediction = model.predict(input_data)

    # Send result back as JSON
    return jsonify({
        "prediction": prediction[0]
    })


if __name__ == "__main__":
    app.run(debug=True)