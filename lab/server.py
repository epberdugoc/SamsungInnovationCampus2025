# Import the Flask class from the flask module
from flask import Flask, request, jsonify
import numpy as np
import logic

# Create an instance of the Flask class, passing in the name of the current module
app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello_world():
    """Function that handles requests to the root URL"""
    return jsonify({"message": "Everything is ready!"}), 200


@app.route("/predictions", methods=["POST"])
def post_predict():
    """Function that handles POST requests to the root URL"""
    input_list = request.get_json()
    results = jsonify({"predictions": logic.predict(np.array(input_list)).tolist()})
    return results


if __name__ == "__main__":
    app.run(debug=False)
