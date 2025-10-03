import numpy as np

train_data = np.load("data/cifar10/all_data/client_0/train_data.npy")
train_targets = np.load("data/cifar10/all_data/client_0/train_targets.npy")
test_data = np.load("data/cifar10/all_data/client_0/test_data.npy")
test_targets = np.load("data/cifar10/all_data/client_0/test_targets.npy")

print("Train data shape:", train_data.shape)
print("Train targets shape:", train_targets.shape)
print("Test data shape:", test_data.shape)
print("Test targets shape:", test_targets.shape)