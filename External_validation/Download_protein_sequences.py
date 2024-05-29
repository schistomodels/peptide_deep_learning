#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:26:08 2023

@author: bioinformatics-14
"""

import pandas as pd
import requests
import os

# Read the Excel file
df = pd.read_excel('Validation_3.xlsx')

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    # Retrieve the protein information from the respective columns
    protein1 = row['Uniprot_accession_number_1']
    protein2 = row['Uniprot_accession_number_2']

    # Create the filename for the first protein
    filename1 = f"{protein1}.fasta"
    # Check if the file already exists in the directory
    if os.path.exists(filename1):
        print(f"Skipping download for {filename1} as it already exists.")
    else:
        # Download the fasta file for the first protein from Uniprot
        r1 = requests.get(f"https://www.uniprot.org/uniprot/{protein1}.fasta")
        # Write the fasta file to disk
        with open(filename1, 'wb') as f1:
            f1.write(r1.content)

    # Create the filename for the second protein
    filename2 = f"{protein2}.fasta"
    # Check if the file already exists in the directory
    if os.path.exists(filename2):
        print(f"Skipping download for {filename2} as it already exists.")
    else:
        # Download the fasta file for the second protein from Uniprot
        r2 = requests.get(f"https://www.uniprot.org/uniprot/{protein2}.fasta")
        # Write the fasta file to disk
        with open(filename2, 'wb') as f2:
            f2.write(r2.content)
