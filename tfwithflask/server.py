from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Initialize global model
global_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

client_weights = []
rounds = 5  # Number of federated learning rounds

def aggregate_weights():
    global global_model
    if client_weights:  # Aggregate only if there are weights from clients
        new_weights = np.mean(client_weights, axis=0)
        global_model.set_weights(new_weights)
        client_weights.clear()
        print("Updated aggregated weights on server")

@app.route('/upload', methods=['POST'])
def upload_weights():
    weights = request.json.get('weights')
    if weights is not None:
        client_weights.append([np.array(w) for w in weights])
        print("Received weights from client")
        return jsonify({"status": "Received weights"})
    return jsonify({"error": "Invalid data"}), 400

@app.route('/get_weights', methods=['GET'])
def get_weights():
    return jsonify({"weights": [w.tolist() for w in global_model.get_weights()]})

@app.route('/train_round', methods=['POST'])
def train_round():
    aggregate_weights()
    return jsonify({"status": "Round completed", "weights": [w.tolist() for w in global_model.get_weights()]})

if __name__ == '__main__':
    print("Initial global model weights on server")
    app.run(host='0.0.0.0', port=5000)
