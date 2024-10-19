from pickle import TRUE
import socket
import time
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

local_weights=np.random.rand(1,7)
local_weights=local_weights[0]

# some basic functions needed for communication.
def string_to_nested_list(data_string):

  try:
    # Attempt to evaluate the string as a list using ast.literal_eval
    data_list = ast.literal_eval(data_string)
  except (SyntaxError, ValueError):
    raise ValueError("Invalid list format in string.")

  # Check if the data is a valid list structure (nested lists)
  if not isinstance(data_list, (list, tuple)):
    raise ValueError("Invalid list format in string.")

  return data_list

def list_to_string(list1):
  return f"[{' '.join(str(x) for x in list1)}]"

 
def updating_weights(list1, list2):
  if len(list1) != len(list2):
    raise ValueError("Lists must be the same size.")

  # Handle 1D lists directly
  if isinstance(list1, (list, tuple)) and len(list1) == 1:
    return [(x + y) / 2 for x, y in zip(list1[0], list2[0])]

  # Recursively handle nested lists
  def average_nested(nested_list1, nested_list2):
    if isinstance(nested_list1, (list, tuple)):
      return [average_nested(sub_list1, sub_list2) for sub_list1, sub_list2 in zip(nested_list1, nested_list2)]
    else:
      return (nested_list1 + nested_list2) / 2

  return average_nested(list1, list2)



def ndarray_to_list_1(arr):
  # Check if the input is a NumPy ndarray
  if not isinstance(arr, np.ndarray):
    raise TypeError("Input must be a NumPy ndarray.")

  # Convert the elements to a list using tolist()
  return arr.tolist()


#model details-model and training-write here all required functions
def linear_regression(x, y):
  # Add a column of ones for the bias term
  X = np.hstack([x, np.ones((x.shape[0], 1))])

  # Calculate the coefficients using linear least squares
  w = np.linalg.inv(X.T @ X) @ (X.T @ y)

  # Return bias (constant term) followed by coefficients
  return np.append(w[0], w[1:])  # Assuming bias is the first element





#main code

serverIp='192.168.161.107'
serverPort=2222
serverAddress=(serverIp,serverPort)
bufferSize=1024
UDPclient=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load the data from the CSV file
file_path = "/content/drive/MyDrive/yasaswini_dataset/crop - crop.csv"
data = pd.read_csv(file_path)

# Assuming your CSV file has columns named 'X' and 'y' for features and target respectively
X = data[['RainFall','Fertilizer','Temperature','Nitrogen','Phosphorus','Potassium']]  # Features
y = data['Yeild']    # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(6,)),  # Input layer with 6 features
    layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    layers.Dense(1)  # Output layer with 1 neuron (for regression) or sigmoid activation (for binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Display the model architecture
model.summary()

start=0
# Train the model
history = model.fit(X_train[start:start+10], y_train[start:start+10], epochs=100, validation_split=0.2, batch_size=32)



# weights is now a list of NumPy arrays containing layer weights
weights = model.get_weights()

for i, weight_array in enumerate(weights):
  print(f"Layer {i+1} Weights Shape:", weight_array.shape)

# Assuming your trained model is stored in 'localweights' variable

start+=10
local_weights=weights
local_weights_str=list_to_string(local_weights)
bytes_to_Send=local_weights_str.encode('utf-8')
UDPclient.sendto(bytes_to_Send,serverAddress)

while True:
    #receiving model updates
    server_data,address = UDPclient.recvfrom(bufferSize)
    server_data=server_data.decode('utf-8')
    print(type(server_data))
    local_weights=string_to_nested_list(server_data)
    model.set_weights(local_weights)
    time.sleep(0.1)
    print("Data from server",local_weights)
    print('client Ip',address[0])
    print('client port',address[1])
    #sending model updates
    #train the model
    history = model.fit(X_train[start:start+10], y_train[start:start+10], epochs=100, validation_split=0.2, batch_size=32)
    weights = model.get_weights()
    start+=10
    local_weights=weights
    local_weights_str=list_to_string(local_weights)
    bytes_to_Send=local_weights_str.encode('utf-8')
    UDPclient.sendto(bytes_to_Send,serverAddress)
    if(start>=2000):
      break

if(start>=2000):
    print("data is over")
else:
   print("model is convergedZ")


loss, mae = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test MAE:", mae)

# Make predictions
predictions = model.predict(X_test)


# Calculate R2 score
r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)

# Calculate root mean squared error (RMSE)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)
