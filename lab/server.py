# Import the Flask class from the flask module
from flask import Flask, request, jsonify
import numpy as np  
import logic

app = Flask(__name__)
print("------- server.py is running -----------")

@app.route("/", methods=['GET'])
def heartbeat():
    """
    Heartbeat endpoint to check if the server is running.
    """
    return jsonify({'message': f"Everything is ready! Tensorflow version: {logic.tf_version()}"}), 200

@app.route('/predictions', methods=['POST'])
def post_predict():

    input_list = request.get_json()

    results = jsonify({'predictions': logic.predict(np.array(input_list)).tolist()})

    return results, 200

if __name__ == '__main__':
    app.run(debug=False)