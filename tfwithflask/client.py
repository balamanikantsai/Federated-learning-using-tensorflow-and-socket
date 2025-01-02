import requests
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Settings
server_url = "http://<server-ip>:5000"  # Replace <server-ip> with Server's IP
rounds = 20  # Number of training rounds

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split the training data into 20 equal parts
x_train_splits = np.array_split(x_train, rounds)
y_train_splits = np.array_split(y_train, rounds)

# Initialize local model
local_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def get_global_weights():
    response = requests.get(f"{server_url}/get_weights")
    if response.status_code == 200:
        weights = response.json().get("weights")
        local_model.set_weights([np.array(w) for w in weights])
        print("Updated global weights received from server")
    else:
        print("Error fetching global weights:", response.json())

def send_local_weights():
    weights = local_model.get_weights()
    response = requests.post(f"{server_url}/upload", json={"weights": [w.tolist() for w in weights]})
    print("Sent local weights to server")
    print("Send weights status:", response.json())

def train_local_model(x_subset, y_subset):
    local_model.fit(x_subset, y_subset, epochs=1, verbose=0)
    print("Local model trained on subset")

def evaluate_model():
    loss, accuracy = local_model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy:", accuracy)
    return accuracy

# Training loop
get_global_weights()  # Get initial global weights
for i in range(rounds):
    print(f"\n--- Round {i+1} ---")

    # Train the model on the subset of data for this round and evaluate accuracy
    train_local_model(x_train_splits[i], y_train_splits[i])
    evaluate_model()

    # Send weights to the server
    send_local_weights()

    # Get aggregated global weights from the server
    get_global_weights()

print("Training completed")
