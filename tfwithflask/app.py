import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import socket
import csv
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import requests
from tensorflow.keras.datasets import mnist

# Global variables
server_process = None
client_thread = None
log_file = None
app = Flask(__name__)

# Server code
global_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

client_weights = []

def aggregate_weights():
    global global_model
    if client_weights:
        new_weights = np.mean(client_weights, axis=0)
        global_model.set_weights(new_weights)
        client_weights.clear()
        log_output.insert(tk.END, "Updated aggregated weights on server\n")
        log_output.see(tk.END)
        log_output.update_idletasks()

@app.route('/upload', methods=['POST'])
def upload_weights():
    weights = request.json.get('weights')
    if weights is not None:
        client_weights.append([np.array(w) for w in weights])
        log_output.insert(tk.END, "Received weights from client\n")
        log_output.see(tk.END)
        log_output.update_idletasks()
        return jsonify({"status": "Received weights"})
    return jsonify({"error": "Invalid data"}), 400

@app.route('/get_weights', methods=['GET'])
def get_weights():
    return jsonify({"weights": [w.tolist() for w in global_model.get_weights()]})

@app.route('/train_round', methods=['POST'])
def train_round():
    aggregate_weights()
    return jsonify({"status": "Round completed", "weights": [w.tolist() for w in global_model.get_weights()]})

def start_server():
    global server_process, log_file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = open(f"server_log_{timestamp}.csv", mode='w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Timestamp", "Weights", "Client IP"])

    def run_server():
        app.run(host='0.0.0.0', port=5000)
    
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    messagebox.showinfo("Server", f"Server started at {get_ip_address()}")

def stop_server():
    global server_process, log_file
    if server_process:
        server_process.terminate()
        server_process = None
        log_file.close()
        messagebox.showinfo("Server", "Server stopped")

# Client code
def get_global_weights(server_url, local_model):
    response = requests.get(f"{server_url}/get_weights")
    if response.status_code == 200:
        weights = response.json().get("weights")
        local_model.set_weights([np.array(w) for w in weights])
        log_output.insert(tk.END, "Updated global weights received from server\n")
        log_output.see(tk.END)
        log_output.update_idletasks()
    else:
        log_output.insert(tk.END, f"Error fetching global weights: {response.json()}\n")
        log_output.see(tk.END)
        log_output.update_idletasks()

def send_local_weights(server_url, local_model):
    weights = local_model.get_weights()
    response = requests.post(f"{server_url}/upload", json={"weights": [w.tolist() for w in weights]})
    log_output.insert(tk.END, "Sent local weights to server\n")
    log_output.insert(tk.END, f"Send weights status: {response.json()}\n")
    log_output.see(tk.END)
    log_output.update_idletasks()

def train_local_model(local_model, x_subset, y_subset):
    local_model.fit(x_subset, y_subset, epochs=1, verbose=0)
    log_output.insert(tk.END, "Local model trained on subset\n")
    log_output.see(tk.END)
    log_output.update_idletasks()

def evaluate_model(local_model, x_test, y_test):
    loss, accuracy = local_model.evaluate(x_test, y_test, verbose=0)
    log_output.insert(tk.END, f"Test accuracy: {accuracy}\n")
    log_output.see(tk.END)
    log_output.update_idletasks()
    return accuracy

def start_client():
    global client_thread, log_file
    server_ip = server_ip_entry.get()
    if not server_ip:
        messagebox.showerror("Error", "Please enter the server IP address")
        return

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = open(f"client_log_{timestamp}.csv", mode='w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Timestamp", "Weights", "Server IP"])

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    rounds = 20
    x_train_splits = np.array_split(x_train, rounds)
    y_train_splits = np.array_split(y_train, rounds)

    local_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def run_client():
        get_global_weights(server_ip, local_model)
        for i in range(rounds):
            log_output.insert(tk.END, f"\n--- Round {i+1} ---\n")
            log_output.see(tk.END)
            log_output.update_idletasks()

            train_local_model(local_model, x_train_splits[i], y_train_splits[i])
            evaluate_model(local_model, x_test, y_test)
            send_local_weights(server_ip, local_model)
            get_global_weights(server_ip, local_model)

        log_output.insert(tk.END, "Training completed\n")
        log_output.see(tk.END)
        log_output.update_idletasks()

    client_thread = threading.Thread(target=run_client)
    client_thread.start()
    messagebox.showinfo("Client", "Client started")

def stop_client():
    global client_thread, log_file
    if client_thread:
        client_thread.join(timeout=1)
        client_thread = None
        log_file.close()
        messagebox.showinfo("Client", "Client stopped")

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# Create the main window
root = tk.Tk()
root.title("Federated Learning App")

# Create buttons to start server and client
btn_server = tk.Button(root, text="Server", command=lambda: show_server_options())
btn_server.pack(pady=10)

btn_client = tk.Button(root, text="Client", command=lambda: show_client_options())
btn_client.pack(pady=10)

log_output = scrolledtext.ScrolledText(root, width=80, height=20)
log_output.pack(pady=10)

def show_server_options():
    server_window = tk.Toplevel(root)
    server_window.title("Server Options")

    btn_start_server = tk.Button(server_window, text="Start Server", command=start_server)
    btn_start_server.pack(pady=10)

    btn_stop_server = tk.Button(server_window, text="Stop Server", command=stop_server)
    btn_stop_server.pack(pady=10)

def show_client_options():
    client_window = tk.Toplevel(root)
    client_window.title("Client Options")

    tk.Label(client_window, text="Server IP:").pack(pady=5)
    global server_ip_entry
    server_ip_entry = tk.Entry(client_window)
    server_ip_entry.pack(pady=5)

    btn_start_client = tk.Button(client_window, text="Start Client", command=start_client)
    btn_start_client.pack(pady=10)

    btn_stop_client = tk.Button(client_window, text="Stop Client", command=stop_client)
    btn_stop_client.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
