# Import the Flask class from the flask module
from flask import Flask, request, jsonify
import numpy as np  

app = Flask(__name__)
print("------- server.py is running -----------")

@app.route("/", methods=['GET'])
def heartbeat():
    """
    Heartbeat endpoint to check if the server is running.
    """
    return jsonify({"message": "Server is running!"}), 200




if __name__ == '__main__':
    app.run(debug=False)