# Federated Learning Implementation

This project demonstrates a federated learning setup using a neural network model. The implementation includes a server and client setup where the server aggregates model weights from multiple clients and updates the global model. The clients train their local models on subsets of the MNIST dataset and send their weights to the server.

## Project Structure

- `server.py`: Sets up a Flask server to handle federated learning. It initializes a global model, aggregates weights from clients, and updates the global model.
- `client.py`: Simulates a client in a federated learning setup. It trains the model on a subset of the MNIST dataset, sends the local model weights to the server, and receives the aggregated global model weights from the server.
- `app.py`: Provides a simple GUI to start and stop the server and client processes. It integrates the functionality of `server.py` and `client.py` directly into the GUI application.
- `bmi.html`: A simple HTML page for calculating Body Mass Index (BMI) with a nice design.

## Requirements

- Python 3.x
- Flask
- TensorFlow
- NumPy
- Requests
- Tkinter (for GUI)
- Scikit-learn
- Pandas

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/federated-learning.git
    cd federated-learning
    ```

2. Install the required packages:
    ```bash
    pip install flask tensorflow numpy requests scikit-learn pandas
    ```

## Usage

### Running the Server

1. Open a terminal and navigate to the project directory.
2. Run the server:
    ```bash
    python server.py
    ```

### Running the Client

1. Open a terminal and navigate to the project directory.
2. Run the client:
    ```bash
    python client.py
    ```

### Using the GUI

1. Open a terminal and navigate to the project directory.
2. Run the GUI application:
    ```bash
    python app.py
    ```
3. Use the GUI to start and stop the server and client processes. The GUI will display logs and updates from the server and client.

### Using the BMI Calculator

1. Open the `bmi.html` file in a web browser.
2. Enter your weight and height, and click the "Calculate BMI" button to see your BMI and category.

## How It Works

### Server

- The server initializes a global model and listens for incoming connections from clients.
- Clients send their local model weights to the server.
- The server aggregates the received weights using the Federated Averaging algorithm and updates the global model.
- The server sends the updated global model weights back to the clients.

### Client

- The client initializes a local model and trains it on a subset of the MNIST dataset.
- The client sends the local model weights to the server.
- The client receives the aggregated global model weights from the server and updates the local model.
- The client repeats the training and weight update process for a specified number of rounds.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.