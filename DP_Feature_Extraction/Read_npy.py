import numpy as np

# Load the numpy array from 'features.npy'
features = np.load('features.npy', allow_pickle=True)

# Print the shape of the array to confirm it loaded correctly
print(features)