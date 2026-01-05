import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

# File path
file_path = r'd:\TUM\TDL\Robust-DPFL-main\Result\gradient_analysis\gradients_FL.json'

def load_and_compute_scores(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return [], []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        malicious_grads = data.get('malicious', [])
        benign_grads = data.get('benign', [])
        
        # Compute L2 Norms
        malicious_scores = [np.linalg.norm(g) for g in malicious_grads]
        benign_scores = [np.linalg.norm(g) for g in benign_grads]
        
        return np.array(benign_scores), np.array(malicious_scores)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

def plot_kde(benign_scores, malicious_scores):
    plt.figure(figsize=(8, 6))
    
    # Determine the range for the x-axis
    all_scores = np.concatenate([benign_scores, malicious_scores])
    if len(all_scores) == 0:
        print("No data to plot.")
        return

    x_min = all_scores.min()
    x_max = all_scores.max()
    
    # Add some padding to the range
    padding = (x_max - x_min) * 0.2
    x_grid = np.linspace(x_min - padding, x_max + padding, 500)
    
    # Plot Benign (Clean)
    if len(benign_scores) > 1:
        kde_benign = gaussian_kde(benign_scores)
        y_benign = kde_benign(x_grid)
        plt.plot(x_grid, y_benign, color='blue', label='Clean')
        plt.fill_between(x_grid, y_benign, color='blue', alpha=0.3)
    else:
        print("Not enough benign data for KDE.")

    # Plot Malicious (Poisoned)
    if len(malicious_scores) > 1:
        kde_malicious = gaussian_kde(malicious_scores)
        y_malicious = kde_malicious(x_grid)
        plt.plot(x_grid, y_malicious, color='red', label='Poisoned')
        plt.fill_between(x_grid, y_malicious, color='red', alpha=0.3)
    else:
        print("Not enough malicious data for KDE.")

    plt.title('Detection Score Distribution (MNIST)')
    plt.xlabel('Detection Score (L2 Norm)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_file = 'mnist_detection_score_kde.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    benign, malicious = load_and_compute_scores(file_path)
    print(f"Benign samples: {len(benign)}, Malicious samples: {len(malicious)}")
    
    plot_kde(benign, malicious)
