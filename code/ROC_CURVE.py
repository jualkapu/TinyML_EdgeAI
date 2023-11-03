import csv
import matplotlib.pyplot as plt

def read_roc_data(filename):
    with open(filename, mode="r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        fpr, tpr, auroc = [float(value) for value in next(reader)]
    return fpr, tpr, auroc

fpr1, tpr1, auroc1 = read_roc_data("model1_roc_data.csv")
fpr2, tpr2, auroc2 = read_roc_data("model2_roc_data.csv")
fpr3, tpr3, auroc3 = read_roc_data("model3_roc_data.csv")

plt.figure()
plt.plot(fpr1, tpr1, label=f"SOM ROC curve (area = {auroc1:.2f})")
plt.plot(fpr2, tpr2, label=f"SAE ROC curve (area = {auroc2:.2f})")
plt.plot(fpr3, tpr3, label=f"UAE ROC curve (area = {auroc3:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
