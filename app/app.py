import os
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# =============================
# 1. Load Pipelines (inside app/)
# =============================
model_A_pipeline = joblib.load(os.path.join("app", "model_A_pipeline.pkl"))
model_B_pipeline = joblib.load(os.path.join("app", "model_B_pipeline.pkl"))

# =============================
# 2. Flask Routes
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_A, prediction_B = None, None

    if request.method == "POST":
        review = request.form["review"]

        # Use pipelines (no need for manual vectorizer transform)
        prediction_A = model_A_pipeline.predict([review])[0]
        prediction_B = model_B_pipeline.predict([review])[0]

    return render_template("index.html",
                           prediction_A=prediction_A,
                           prediction_B=prediction_B)


if __name__ == "__main__":
    app.run(debug=True)

