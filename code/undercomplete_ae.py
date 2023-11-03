import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# Define a data generator function for incremental learning
def data_generator(data, batch_size=10):
    i = 0
    while True:
        batch = data[i:i+batch_size, :]
        i += batch_size
        if i >= data.shape[0]:
            i = 0
        yield batch, batch

# Load the data from CSV file
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

# Separate the labels from the features
X_train = train_data.values[:, :-1].astype(np.float32)
X_test = test_data.values[:, :-1].astype(np.float32)
y_test = test_data.values[:, -1].astype(int)

# Remove rows with NaN values
nan_rows = np.isnan(X_test).any(axis=1)
X_test = X_test[nan_rows == False]
y_test = y_test[nan_rows == False]

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the dimensions of the encoder and decoder layers
input_dim = X_train.shape[1]
encoding_dim = 250

# Define the input layer
input_layer = Input(shape=(input_dim,))

# Define the encoder layer
encoder_layer = Dense(encoding_dim, activation='LeakyReLU')(input_layer)

# Define the decoder layer
decoder_layer = Dense(input_dim, activation='LeakyReLU')(encoder_layer)

# Define the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder_layer)

# Define the encoder model
encoder = Model(inputs=input_layer, outputs=encoder_layer)

# Define the decoder model
decoder_input = Input(shape=(encoding_dim,))
decoder_output = autoencoder.layers[-1](decoder_input)
decoder = Model(inputs=decoder_input, outputs=decoder_output)

# Compile the autoencoder model
lr = 1e-5  # Initial learning rate
optimizer = keras.optimizers.Adam(lr=lr)
autoencoder.compile(optimizer=optimizer, loss=keras.losses.Huber())

# Train the autoencoder model incrementally using a data generator
batch_size = 10
train_generator = data_generator(X_train, batch_size=batch_size)
steps_per_epoch = len(X_train) // batch_size
start_time = time.time()
for i, (batch, _) in enumerate(train_generator):
    autoencoder.train_on_batch(batch, batch)
    if i % 500 == 0:
        print(f"Processed {i*batch_size} samples in {time.time()-start_time:.2f} seconds")
    if i >= steps_per_epoch:
        break

print(f"Model training time: {time.time() - start_time:.2f}s")

# Use the encoder to get the encoded representation of the test data
encoded_test = encoder.predict(X_test)

# Use the decoder to get the reconstructed test data
reconstructed_test = decoder.predict(encoded_test)

# Find the reconstruction error for each test data point
mse = np.mean(np.power(X_test - reconstructed_test, 2), axis=1)

# Find the threshold error based on the 85th percentile of the reconstruction errors
threshold = np.percentile(mse, 75)

# Classify the testing points as normal (0) or anomalous (1) based on the threshold
y_pred = np.zeros_like(y_test)
y_pred[mse > threshold] = 1

# Evaluate the performance of the model
# Evaluate the performance of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, mse)  # Calculate AUROC score

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"AUROC: {auroc:.2f}")  # Print AUROC score

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, mse)
plt.plot(fpr, tpr, label=f'AUROC: {auroc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# Convert the model to tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

# Save the tflite model
with open('sparse_uae.tflite', 'wb') as f:
    f.write(tflite_model)

# Measure the size of the tflite model
tflite_model_size = len(tflite_model) / 1024 # convert from bytes to kilobytes

print(f"Size of TFLite model: {tflite_model_size:.2f} KB")