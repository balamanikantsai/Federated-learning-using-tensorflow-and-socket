
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Initialize global weights as a numpy array
global_weights = None
client_weights = []
rounds = 5  # Number of federated learning rounds

def aggregate_weights():
    global global_weights
    if client_weights:  # Aggregate only if there are weights from clients
        global_weights = np.mean(client_weights, axis=0)
        client_weights.clear()
        print("Updated aggregated weights on server:", global_weights)

@app.route('/upload', methods=['POST'])
def upload_weights():
    weights = request.json.get('weights')
    if weights is not None:
        client_weights.append(np.array(weights))
        print("Received weights from client:", weights)
        return jsonify({"status": "Received weights"})
    return jsonify({"error": "Invalid data"}), 400

@app.route('/get_weights', methods=['GET'])
def get_weights():
    if global_weights is None:
        return jsonify({"error": "No global weights available"}), 404
    return jsonify({"weights": global_weights.tolist()})

@app.route('/train_round', methods=['POST'])
def train_round():
    aggregate_weights()
    if global_weights is None:
        return jsonify({"error": "No global weights available"}), 404
    return jsonify({"status": "Round completed", "weights": global_weights.tolist()})

if __name__ == '__main__':
    # Initialize global weights with random values to start federated learning
    global_weights = np.random.rand(4)  # Adjust the size to match feature importances
    print("Initial global weights on server:", global_weights)
    app.run(host='0.0.0.0', port=5000)
