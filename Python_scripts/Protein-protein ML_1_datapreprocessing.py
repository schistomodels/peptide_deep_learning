# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:51:32 2023

@author: K. Agyenkwa-Mawuli
"""

import numpy as np
import pandas as pd
import os
import random


# Define hyperparameters
maxlen = 1024  # maximum sequence length
fasta_dir = r'C:/Users/K. Agyenkwa-Mawuli/Desktop/Santa'
worksheet_path = r'C:/Users/K. Agyenkwa-Mawuli/Desktop/Santa/worksheet.xlsx' 
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
data = load_data(worksheet_path, fasta_dir, labels_col)

# Shuffle the data
random.shuffle(data)


# Split the data into training, validation, and testing sets
train_size = int(0.7 * len(data))
val_size = int(0.1 * len(data))
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Print the sizes of the datasets
print("Training data size:", len(train_data))
print("Validation data size:", len(val_data))
print("Testing data size:", len(test_data))


# Extract the sequences and labels from the train_data and val_data
train_sequences1 = np.array([pair[0] for pair, label in train_data])
train_sequences2 = np.array([pair[1] for pair, label in train_data])
train_labels = np.array([label for pair, label in train_data])

val_sequences1 = np.array([pair[0] for pair, label in val_data])
val_sequences2 = np.array([pair[1] for pair, label in val_data])
val_labels = np.array([label for pair, label in val_data])

test_sequences1 = np.array([pair[0] for pair, label in test_data])
test_sequences2 = np.array([pair[1] for pair, label in test_data])
test_labels = np.array([label for pair, label in test_data])

# Define the file paths for saving the arrays
train_sequences1_path = 'train_sequences1.npy'
train_sequences2_path = 'train_sequences2.npy'
train_labels_path = 'train_labels.npy'

val_sequences1_path = 'val_sequences1.npy'
val_sequences2_path = 'val_sequences2.npy'
val_labels_path = 'val_labels.npy'

test_sequences1_path = 'test_sequences1.npy'
test_sequences2_path = 'test_sequences2.npy'
test_labels_path = 'test_labels.npy'

# Save the arrays
np.save(train_sequences1_path, train_sequences1)
np.save(train_sequences2_path, train_sequences2)
np.save(train_labels_path, train_labels)

np.save(val_sequences1_path, val_sequences1)
np.save(val_sequences2_path, val_sequences2)
np.save(val_labels_path, val_labels)

np.save(test_sequences1_path, test_sequences1)
np.save(test_sequences2_path, test_sequences2)
np.save(test_labels_path, test_labels)
