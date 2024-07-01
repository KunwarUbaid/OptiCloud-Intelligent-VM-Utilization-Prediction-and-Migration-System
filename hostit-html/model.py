import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import LSTM

# Load the dataset into a pandas DataFrame
df = pd.read_csv(r'D:\cpu_data (1).csv', skiprows=1)

# Extract the input features and the target variable
y = df.iloc[-1, :].values

X = []
for col in df.columns:
    data = df[col].to_numpy()
    X.append(data[:287])
X = np.array(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model with output shape matching LSTM input shape
nn_model = Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(287, activation='linear')  # Output shape matching LSTM input
])

# ... Rest of the code remains the same


# Build the LSTM model
lstm_model = Sequential([
    LSTM(100, input_shape=(X_train.shape[1], 1)),
    layers.Dense(1)
])

# Reshape the input for the LSTM
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Combine the neural network and LSTM models
combined_model = Sequential()
combined_model.add(nn_model)
combined_model.add(lstm_model)

# Compile the combined model
combined_model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mae', 'mse'])

# Train the model on the training set
history = combined_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32,
                             validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test), verbose=0)

# Predict the target variable for the testing set using the trained model
y_pred = combined_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(combined_model, model_file)
    
# Load the saved model from the .pkl file
with open('model.pkl', 'rb') as model_file:
    combined_model = pickle.load(model_file)