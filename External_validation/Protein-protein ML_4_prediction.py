# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:51:32 2023

@author: K. Agyenkwa-Mawuli
"""

import numpy as np
import pandas as pd
import os
import random
from tensorflow.keras.models import load_model


# Define hyperparameters
maxlen = 1024  # maximum sequence length
fasta_dir = r'./'
worksheet_path = r'./Validation_3.xlsx' 
labels_col = 'Label'

# Define the mapping from amino acid characters to integers
charmap = {}
index = 1

# Generate keys for AAA to YYY
for i in range(ord('A'), ord('Y')+1):
    for j in range(ord('A'), ord('Y')+1):
        for k in range(ord('A'), ord('Y')+1):
            triplet = chr(i) + chr(j) + chr(k)
            charmap[triplet] = index
            index += 1

# Add padding key with value 0
charmap['PAD'] = 0

   
def load_sequences(fasta_path):
    with open(fasta_path, 'r') as file:
        lines = file.readlines()
        sequence = ''
        for line in lines[1:]:
            if line.startswith('>'):
                continue
            sequence += line.strip()
        seq = np.array([charmap.get(sequence[i:i+3], 0) for i in range(0, len(sequence), 3)], dtype=np.int32)
        if len(seq) > maxlen:
            seq = seq[-maxlen:]
        else:
            seq = np.pad(seq, (maxlen - len(seq), 0), mode='constant', constant_values=0)
    return seq
     
def load_data(worksheet_path, fasta_dir, labels_col):
    # Load interaction data from worksheet
    df = pd.read_excel(worksheet_path)
    data = []
    # Iterate over rows in the worksheet
    for i, row in df.iterrows():
        # Get the fasta file paths for the protein sequences
        fasta1_path = os.path.join(fasta_dir, row['Uniprot_accession_number_1'] + '.fasta')
        fasta2_path = os.path.join(fasta_dir, row['Uniprot_accession_number_2'] + '.fasta')
        # Check if fasta files exist
        if not (os.path.exists(fasta1_path) and os.path.exists(fasta2_path)):
            continue
        # Load protein sequence pairs from fasta files
        sequences1 = load_sequences(fasta1_path)
        sequences2 = load_sequences(fasta2_path)
        label = row[labels_col]
        data.append(((sequences1, sequences2), label))
    return data

      
# Load protein sequence pairs and interaction labels
test_data = load_data(worksheet_path, fasta_dir, labels_col)


# Print the sizes of the datasets
print("Testing data size:", len(test_data))


# Extract the sequences and labels from the train_data and val_data
test_sequences1 = np.array([pair[0] for pair, label in test_data])
test_sequences2 = np.array([pair[1] for pair, label in test_data])
test_labels = np.array([label for pair, label in test_data])

# Define the file paths for saving the arrays
test_sequences1_path = 'test_sequences1.npy'
test_sequences2_path = 'test_sequences2.npy'
test_labels_path = 'test_labels.npy'

# Save the arrays
np.save(test_sequences1_path, test_sequences1)
np.save(test_sequences2_path, test_sequences2)
np.save(test_labels_path, test_labels)

# Load model 
model = load_model("PPI_model.h5")

# Make predictions on the test data
test_predictions = model.predict([test_sequences1, test_sequences2])

# Convert probability predictions to binary predictions
test_predictions_binary = (test_predictions > 0.5).astype(int)

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame(np.hstack((test_predictions, test_predictions_binary)),
                              columns=["test_predictions", "test_predictions_binary"])

# Print the predictions as columns
print(predictions_df)

# Save the DataFrame as an Excel file
predictions_df.to_excel("predictions_table.xlsx", index=False)

# Count the number of '1s' in the predictions_binary column
num_ones = predictions_df["test_predictions_binary"].sum()
print("Number of '1s' in predictions_binary column:", num_ones)