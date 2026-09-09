from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

DATA_PATH = "/workspaces/XO10_DS02/development_train.csv"


@app.route("/")
def home():
    return jsonify({
        "message": "SENTINEL API is running",
        "status": "online"
    })


@app.route("/api/summary")
def summary():

    df = pd.read_csv(DATA_PATH)

    return jsonify({
        "total_records": len(df),
        "stations": int(df["Station_ID"].nunique()),
        "sensors": 6
    })


@app.route("/api/readings")
def readings():

    df = pd.read_csv(DATA_PATH)

    data = df.head(20).replace(
        {np.nan: None}
    ).to_dict(orient="records")

    return jsonify(data)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )