# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:51:32 2023

@author: K. Agyenkwa-Mawuli
"""

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf 

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

   
def preprocess_sequences(input_data1, input_data2): 
    def load_sequences(input_data): 
        with open(input_data, 'r') as file: 
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
 
    # Load sequence 1 
    sequences1 = load_sequences(input_data1) 
 
    # Load sequence 2 
    sequences2 = load_sequences(input_data2) 
 
    return sequences1[np.newaxis, :], sequences2[np.newaxis, :]

destination_1 = r'C:/Users/K. Agyenkwa-Mawuli/Desktop/Santa/P17382.fasta' 
destination_2 = r'C:/Users/K. Agyenkwa-Mawuli/Desktop/Santa/P0CB38.fasta'
sequences1, sequences2 = preprocess_sequences(destination_1, destination_2) 

sequences1 = np.array(sequences1)
sequences2 = np.array(sequences2)

# Load the deep learning model 
model = tf.keras.models.load_model("PPI_best_final.h5") 

# Make predictions on the preprocessed sequences 
test_predictions = model.predict([sequences1, sequences2])

# Convert probability predictions to binary predictions
test_predictions_binary = (test_predictions > 0.5).astype(int)
print(test_predictions)
print(test_predictions_binary)