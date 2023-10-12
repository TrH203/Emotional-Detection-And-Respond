import numpy as np

k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

# Split the training data into folds

# Initialize a dictionary to store accuracy scores for each k value
k_to_accuracies = {}

# Iterate over each k value
for k in k_choices:
    k_to_accuracies[k] = []