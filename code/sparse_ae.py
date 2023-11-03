# Import necessary libraries
from tensorflow import keras # for building and training neural networks
import numpy as np # for numerical computing in Python
from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn import metrics # for evaluating the performance of the model
import tensorflow as tf # for converting to tflite and measuring size
import time
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

# Load and preprocess data
train_data = np.loadtxt('train.csv', delimiter=',') # load training data from CSV file
test_data = np.loadtxt('test.csv', delimiter=',') # load test data from CSV file
X_train = train_data[:, :-1].astype(np.float32) # extract training features and convert to float32 data type
X_test = test_data[:, :-1].astype(np.float32) # extract test features and convert to float32 data type
y_test = test_data[:, -1].astype(int) # extract test labels and convert to integer data type

# Scale the data
scaler = StandardScaler() # create a scaler object to scale the data
X_train = scaler.fit_transform(X_train) # fit the scaler to the training data and transform the data
X_test = scaler.transform(X_test) # transform the test data using the same scaler

# Define the autoencoder model
model = keras.models.Sequential([
    keras.layers.Dense(150, activation="LeakyReLU", input_shape=X_train.shape[1:]),
    keras.layers.Dense(5, activation="LeakyReLU", activity_regularizer=keras.regularizers.l1(10e-5)), # add sparsity constraint to bottleneck layer
    keras.layers.Dense(150, activation="LeakyReLU"),
    keras.layers.Dense(X_train.shape[1])
])

# Compile the model
model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=1e-3)) # specify the loss function (mean squared error) and the optimizer (Adam) with a learning rate of 0.001

# Train the model incrementally using a data generator
start_time = time.time()  # Initialize start_time
batch_size = 1
steps_per_epoch = X_train.shape[0] // batch_size
train_generator = data_generator(X_train, batch_size=batch_size)
for i, (batch, _) in enumerate(train_generator):
    loss = model.train_on_batch(batch, batch)
    if i % 1000 == 0:
        print(f"Batch {i+1}/{steps_per_epoch} - Loss: {loss:.4f}, time elapsed: {time.time() - start_time:.2f}s")
    if i >= steps_per_epoch:
        break

print(f"Model training time: {time.time() - start_time:.2f}s")

# Predict the reconstruction error on the test set
y_pred = model.predict(X_test) # use the trained model to make predictions on the test set
mse = np.mean(np.power(X_test - y_pred, 2), axis=1) # calculate the mean squared error between the test set and the reconstructed data, and take the mean across features

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 75) # set the threshold for anomaly detection to the 85th percentile of the mean squared error values

# Classify the testing points as normal (0) or anomalous (1) based on the threshold
y_pred = np.zeros_like(y_test) # initialize the predicted labels as all zeros
y_pred[mse > threshold] = 1 # classify the testing points with mean squared error values greater than the threshold as anomalous (1), and leave the others as normal (0)

# Evaluate the performance of the model
accuracy = metrics.accuracy_score(y_test, y_pred) # calculate the accuracy of the model
precision = metrics.precision_score(y_test, y_pred) # calculate the precision of the model
recall = metrics.recall_score(y_test, y_pred) # calculate the recall of the model
f1_score = metrics.f1_score(y_test, y_pred) # calculate the F1 score of the model
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
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the tflite model
with open('sparse_ae.tflite', 'wb') as f:
    f.write(tflite_model)

# Measure the size of the tflite model
tflite_model_size = len(tflite_model) / 1024 # convert from bytes to kilobytes

print(f"Size of TFLite model: {tflite_model_size:.2f} KB")
