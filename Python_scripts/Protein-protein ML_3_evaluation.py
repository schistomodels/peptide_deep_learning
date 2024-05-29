# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:51:32 2023

@author: K. Agyenkwa-Mawuli
"""

import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score


# Define the file paths for the saved arrays
test_sequences1_path = 'test_sequences1.npy'
test_sequences2_path = 'test_sequences2.npy'
test_labels_path = 'test_labels.npy'

# Load the arrays from the saved files
test_sequences1 = np.load(test_sequences1_path)
test_sequences2 = np.load(test_sequences2_path)
test_labels = np.load(test_labels_path)


# Load model 
model = load_model("PPI_best_final.h5")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate([test_sequences1, test_sequences2], test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

test_predictions = model.predict([test_sequences1, test_sequences2])

# Convert probability predictions to binary predictions
test_predictions_binary = (test_predictions > 0.5).astype(int)

# Calculate evaluation metrics
balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions_binary)
precision = precision_score(test_labels, test_predictions_binary)
recall = recall_score(test_labels, test_predictions_binary)
specificity = recall_score(test_labels, test_predictions_binary, pos_label=0)
mcc = matthews_corrcoef(test_labels, test_predictions_binary)
f1 = f1_score(test_labels, test_predictions_binary)
roc_auc = roc_auc_score(test_labels, test_predictions)

# Display the evaluation metrics
print("Balanced Accuracy:", balanced_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("Matthew's Correlation Coefficient (MCC):", mcc)
print("F1-score:", f1)
print("ROC-AUC score:", roc_auc)





