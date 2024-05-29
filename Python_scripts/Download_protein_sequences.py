# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:50:20 2023

@author: K. Agyenkwa-Mawuli
"""

#!/usr/bin/env python
import requests
import os

# Open the worksheet file
with open('worksheet.tsv', 'r') as f:
    # Skip the header row
    next(f)
    # Loop through the remaining rows
    for line in f:
        # Strip the newline character
        line = line.strip()
        # Split the line into columns
        cols = line.split('\t')
        # Create the filename for the first protein
        filename1 = f"{cols[1]}.fasta"
        # Check if the file already exists in the directory
        if os.path.exists(filename1):
            print(f"Skipping download for {filename1} as it already exists.")
        else:
            # Download the fasta file for the first protein from Uniprot
            r1 = requests.get(f"https://www.uniprot.org/uniprot/{cols[1]}.fasta")
            # Write the fasta file to disk
            with open(filename1, 'wb') as f1:
                f1.write(r1.content)
        # Create the filename for the second protein
        filename2 = f"{cols[3]}.fasta"
        # Check if the file already exists in the directory
        if os.path.exists(filename2):
            print(f"Skipping download for {filename2} as it already exists.")
        else:
            # Download the fasta file for the second protein from Uniprot
            r2 = requests.get(f"https://www.uniprot.org/uniprot/{cols[2]}.fasta")
            # Write the fasta file to disk
            with open(filename2, 'wb') as f2:
                f2.write(r2.content)

