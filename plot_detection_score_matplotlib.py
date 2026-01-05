import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import os

# File path
file_path = r'd:\TUM\TDL\Robust-DPFL-main\Result\gradient_analysis\gradients_last_round.json'

def load_and_compute_scores(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return [], []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        malicious_grads = data.get('malicious', [])
        benign_grads = data.get('benign', [])
        
        # Compute Detection Scores: abs(mean(vector))
        malicious_scores = [np.abs(np.mean(g)) for g in malicious_grads]
        benign_scores = [np.abs(np.mean(g)) for g in benign_grads]
        
        return np.array(benign_scores), np.array(malicious_scores)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

def plot_kde(benign_scores, malicious_scores):
    # Clean data: remove NaNs and Infs
    benign_scores = benign_scores[np.isfinite(benign_scores)]
    malicious_scores = malicious_scores[np.isfinite(malicious_scores)]

    plt.figure(figsize=(8, 6))
    
    # Determine the range for the x-axis
    all_scores = np.concatenate([benign_scores, malicious_scores])
    if len(all_scores) == 0:
        print("No data to plot.")
        return

    x_min = all_scores.min()
    x_max = all_scores.max()
    
    # Add some padding to the range
    padding = (x_max - x_min) * 0.5
    x_grid = np.linspace(x_min - padding, x_max + padding, 1000)
    
    # Plot Benign (Clean) - Blue
    if len(benign_scores) > 1:
        kde_benign = gaussian_kde(benign_scores)
        y_benign = kde_benign(x_grid)
        plt.plot(x_grid, y_benign, color='blue', label='Clean')
        plt.fill_between(x_grid, y_benign, color='blue', alpha=0.3)
    
    # Plot Malicious (Poisoned) - Red
    if len(malicious_scores) > 1:
        kde_malicious = gaussian_kde(malicious_scores)
        y_malicious = kde_malicious(x_grid)
        plt.plot(x_grid, y_malicious, color='red', label='Poisoned')
        plt.fill_between(x_grid, y_malicious, color='red', alpha=0.3)
    
    plt.title('CIFAR10 Detection Score Distribution (Last Round)')
    plt.xlabel('Detection Score')
    plt.ylabel('Density')
    
    # Force X-axis to use scientific notation (10^-4)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-4, -4))
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # Set axes limits
    plt.xlim(left=x_min - padding, right=x_max + padding)
    plt.ylim(bottom=0)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_file = 'detection_score_matplotlib.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    benign, malicious = load_and_compute_scores(file_path)
    
    if len(benign) == 0 and len(malicious) == 0:
        print("No data found.")
    else:
        print(f"Benign samples: {len(benign)}, Malicious samples: {len(malicious)}")
        plot_kde(benign, malicious)
