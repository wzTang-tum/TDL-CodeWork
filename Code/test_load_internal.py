
from preprocessing import load_femnist_data
import os

print(f"CWD: {os.getcwd()}")
print("Loading FEMNIST data...")
try:
    train_images, train_labels, test_images, test_labels = load_femnist_data()
    print("Load complete!")
    print(f"Train images shape: {train_images.shape}")
except Exception as e:
    print(f"Error: {e}")
