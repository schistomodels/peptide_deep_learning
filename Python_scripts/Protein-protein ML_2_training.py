# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:51:32 2023

@author: K. Agyenkwa-Mawuli
"""

import numpy as np


from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model



# Define hyperparameters
batch_size = 64
epochs = 25
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


# Define the file paths for the saved arrays
train_sequences1_path = 'train_sequences1.npy'
train_sequences2_path = 'train_sequences2.npy'
train_labels_path = 'train_labels.npy'

val_sequences1_path = 'val_sequences1.npy'
val_sequences2_path = 'val_sequences2.npy'
val_labels_path = 'val_labels.npy'

test_sequences1_path = 'test_sequences1.npy'
test_sequences2_path = 'test_sequences2.npy'
test_labels_path = 'test_labels.npy'

# Load the arrays from the saved files
train_sequences1 = np.load(train_sequences1_path)
train_sequences2 = np.load(train_sequences2_path)
train_labels = np.load(train_labels_path)

val_sequences1 = np.load(val_sequences1_path)
val_sequences2 = np.load(val_sequences2_path)
val_labels = np.load(val_labels_path)

test_sequences1 = np.load(test_sequences1_path)
test_sequences2 = np.load(test_sequences2_path)
test_labels = np.load(test_labels_path)


# Define the model architecture

# Define the input layers for sequence1 and sequence2
input1 = Input(shape=(maxlen,))
input2 = Input(shape=(maxlen,))

# Embedding layer for sequence1
embedding1 = Embedding(input_dim=len(charmap) + 1, output_dim=512)(input1)
# Batch normalization layer for embedding1
bn1 = BatchNormalization()(embedding1)
# Recurrent layer for embedding1
lstm1 = LSTM(64, return_sequences=False)(bn1)
# Fully connected layer for lstm1
fc1 = Dense(64)(lstm1)
# Batch normalization layer for fc1
bn2 = BatchNormalization()(fc1)

# Embedding layer for sequence2
embedding2 = Embedding(input_dim=len(charmap) + 1, output_dim=512)(input2)
# Batch normalization layer for embedding2
bn3 = BatchNormalization()(embedding2)
# Recurrent layer for embedding2
lstm2 = LSTM(64, return_sequences=False)(bn3)
# Fully connected layer for lstm2
fc2 = Dense(64)(lstm2)
# Batch normalization layer for fc2
bn4 = BatchNormalization()(fc2)

# Concatenate the output of fc1 and fc2
concatenated = Concatenate()([bn2, bn4])

# Fully connected layer for concatenated
fc3 = Dense(512)(concatenated)
# Batch normalization layer for fc3
bn5 = BatchNormalization()(fc3)

# Sigmoid output layer
output = Dense(1, activation='sigmoid')(bn5)

# Define the model
model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model

# Define the model checkpoint callback
checkpoint = ModelCheckpoint("PPI_best_final.h5",
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)


model.fit([train_sequences1, train_sequences2], train_labels, 
          validation_data=([val_sequences1, val_sequences2], val_labels), 
          batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])

model = load_model("PPI_best_final.h5")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate([test_sequences1, test_sequences2], test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)





