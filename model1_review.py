# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the data from the CSV file
file_path = "/content/drive/MyDrive/yasaswini_dataset/crop - crop.csv"
data = pd.read_csv(file_path)

# Assuming your CSV file has columns named 'X' and 'y' for features and target respectively
X = data[['RainFall','Fertilizer','Temperature','Nitrogen','Phosphorus','Potassium']]  # Features
y = data['Yeild']    # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the neural network architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(6,)),  # Input layer with 6 features
    layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
    layers.Dense(1)  # Output layer with 1 neuron (for regression) or sigmoid activation (for binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Display the model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test MAE:", mae)

# Make predictions
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
print("R2 Score:", r2)

# Calculate root mean squared error (RMSE)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)
