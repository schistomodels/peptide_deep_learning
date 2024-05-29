# -*- coding: utf-8 -*-
"""
Created on Tue June 13 14:42:32 2023

@author: K. Agyenkwa-Mawuli
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model



    
    
# Define the mapping from amino acid characters to integers
charmap = {}
index = 1
maxlen = 1024  # maximum sequence length

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
            lines = input_data.read().decode("utf-8").splitlines()
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

    return sequences1, sequences2

# Streamlit app
def main():
    st.title('Protein Sequence Preprocessing and Prediction')
    st.write('Upload two files for sequence 1 and sequence 2:')
    uploaded_file1 = st.file_uploader('Choose file for sequence 1', type=['fasta'])
    uploaded_file2 = st.file_uploader('Choose file for sequence 2', type=['fasta'])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        sequences1, sequences2 = preprocess_sequences(uploaded_file1, uploaded_file2)

        # Load the deep learning model
        model = load_model("PPI_model.h5")
    
    # Make predictions on the preprocessed sequences
        predictions = model.predict([sequences1, sequences2])

        st.write('Preprocessed Sequences:')
        st.write('Sequence 1:')
        st.write(sequences1)
        st.write('Sequence 2:')
        st.write(sequences2)
        st.write('Predictions:')
        st.write(predictions)

if __name__ == '__main__':
    main()
    
