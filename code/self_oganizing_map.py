import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from minisom import MiniSom
import time
import pickle
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# Load the data from CSV file
train_data = pd.read_csv('train.csv', header=None)

# Separate the labels from the features
X_train = train_data.values[:, :-1].astype(np.float32)
y_train = train_data.values[:, -1].astype(int)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Initialize the SOM model object
som = MiniSom(x=21, y=21, input_len=X_train.shape[1], sigma=0.5, learning_rate=0.4, random_seed=123)


# Save the SOM model to a file
with open('som_model.pickle', 'wb') as f:
    pickle.dump(som, f)

# Load the saved SOM model
with open('som_model.pickle', 'rb') as f:
    som = pickle.load(f)

# Incrementally update the SOM with new training data
start_time = time.time()  # Initialize start_time
batch_size = 10
for i in range(batch_size, len(X_train), batch_size):
    batch = X_train[i:i+batch_size]
    som.train_batch(batch, num_iteration=10)

    # Save the SOM model to a file periodically
    if i % 1000 == 0:
        with open('som_model.pickle', 'wb') as f:
            pickle.dump(som, f)
        
    # Log progress every 100 iterations
    if i % 10000 == 0:
        print(f"Iteration {i}/{len(X_train)}, time elapsed: {time.time() - start_time:.2f}s")

print(f"Model training time: {time.time() - start_time:.2f}s")


# Load the test data
test_data = pd.read_csv('test.csv', header=None)

# Separate the labels from the features
X_test = test_data.values[:, :-1].astype(np.float32)
y_test = test_data.values[:, -1].astype(int)

# Remove rows with NaN values
nan_rows = np.isnan(X_test).any(axis=1)
X_test = X_test[nan_rows == False]
y_test = y_test[nan_rows == False]

# Scale the data
X_test = scaler.transform(X_test)

# Find the distances of all testing points to the closest SOM node
distances = np.zeros(len(X_test))
for i, x in enumerate(X_test):
    w = som.winner(x)
    distances[i] = np.linalg.norm(x - som.get_weights()[w])

# Find the threshold distance based on the 85th percentile of the distances
threshold = np.percentile(distances, 70)

# Classify the testing points as normal (0) or anomalous (1) based on the threshold
y_pred = np.zeros_like(y_test)
y_pred[distances > threshold] = 1

# Evaluate the performance of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, distances)  # Calculate AUROC score

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"AUROC: {auroc:.2f}")  # Print AUROC score

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, distances)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {auroc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# Measure the size of the SOM model
som_model_size = os.path.getsize('som_model.pickle') / 1024 # convert from bytes to kilobytes
print(f"Size of SOM model: {som_model_size:.2f} KB")