import requests
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Settings
server_url = "http://<server-ip>:5000"  # Replace <server-ip> with Server's IP
local_model = DecisionTreeClassifier()
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
rounds = 20  # Number of training rounds

# Split the training data into 20 equal parts
X_train_splits = np.array_split(X_train, rounds)
y_train_splits = np.array_split(y_train, rounds)

def get_global_weights():
    response = requests.get(f"{server_url}/get_weights")
    if response.status_code == 200:
        try:
            weights = np.array(response.json().get("weights"))
            print("Updated global weights received from server:", weights)
            return weights
        except ValueError as e:
            print("JSON decode error:", e)
            return None
    else:
        print("Error fetching global weights:", response.json())
    return None

def send_local_weights(weights):
    response = requests.post(f"{server_url}/upload", json={"weights": weights.tolist()})
    print("Sent local weights to server:", weights)
    print("Send weights status:", response.json())

def train_local_model(X_subset, y_subset):
    local_model.fit(X_subset, y_subset)
    # Extract the feature importances as weights for the DecisionTree
    local_weights = local_model.feature_importances_
    print("Local weights after training:", local_weights)
    return local_weights

def evaluate_model():
    y_pred = local_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy:", accuracy)
    return accuracy

# Training loop
for i in range(rounds):
    print(f"\n--- Round {i+1} ---")

    # Train the model on the subset of data for this round and evaluate accuracy
    local_weights = train_local_model(X_train_splits[i], y_train_splits[i])
    accuracy = evaluate_model()

    # Send weights to the server
    send_local_weights(local_weights)

    # Get aggregated global weights from the server
    global_weights = get_global_weights()
    if global_weights is not None:
        print(f"Using updated global weights for round {i+1}: {global_weights}")

print("Training completed")
