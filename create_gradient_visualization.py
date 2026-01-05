import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def visualize_gradients_2d(json_path, output_path=None):
    """
    Create 2D visualization of gradients as described in the paper:
    "Gradient distributions of conventional FL and DPFL. Gradients in the gray circle
    are aggregated for global model updating in previous robust aggregation methods."
    """

    # Load gradient data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract gradients
    malicious_grads = np.array(data['malicious'])  # Shape: (num_malicious, gradient_size)
    benign_grads = np.array(data['benign'])       # Shape: (num_benign, gradient_size)
    
    # Handle selected gradient (which might be a list of layers with varying shapes)
    selected_raw = data['selected']
    flat_selected = []
    
    if isinstance(selected_raw, list):
        for item in selected_raw:
            # item could be a scalar, a list (1D layer), or a nested list (multi-D layer)
            # Using np.array(item).flatten() handles all these cases robustly
            flat_item = np.array(item).flatten()
            flat_selected.extend(flat_item)
    else:
        # It's just a scalar or single array
        flat_selected = np.array(selected_raw).flatten()
        
    selected_grad = np.array(flat_selected)
        
    # Ensure selected_grad has the same size as other gradients
    target_size = malicious_grads.shape[1] if len(malicious_grads) > 0 else (benign_grads.shape[1] if len(benign_grads) > 0 else 0)
    
    if target_size > 0 and selected_grad.size != target_size:
        # Trim or pad to match
        if selected_grad.size > target_size:
            selected_grad = selected_grad[:target_size]
        else:
            # Pad with zeros if smaller (unlikely given the logic, but safe)
            selected_grad = np.pad(selected_grad, (0, target_size - selected_grad.size))

    print(f"Loaded {len(malicious_grads)} malicious and {len(benign_grads)} benign gradients")
    print(f"Each gradient has {malicious_grads.shape[1]} elements")

    # Combine all client gradients for PCA
    all_grads = np.vstack([malicious_grads, benign_grads])
    labels = ['malicious'] * len(malicious_grads) + ['benign'] * len(benign_grads)

    # Perform PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    grads_2d = pca.fit_transform(all_grads)

    # Split back into malicious and benign
    malicious_2d = grads_2d[:len(malicious_grads)]
    benign_2d = grads_2d[len(malicious_grads):]

    # Perform K-means clustering to identify the selected cluster (like RobustDPFL)
    # Use the same clustering logic as in the paper
    z_scores = np.abs(all_grads.mean(axis=-1))  # Same as RobustAggergation
    kmeans = KMeans(n_clusters=2, random_state=9).fit(z_scores.reshape(-1, 1))
    cluster_labels = kmeans.predict(z_scores.reshape(-1, 1))

    # Determine which cluster was selected (the one with smaller mean z-score)
    cluster_means = [z_scores[cluster_labels == i].mean() for i in range(2)]
    selected_cluster = np.argmin(cluster_means)
    selected_indices = np.where(cluster_labels == selected_cluster)[0]

    print(f"Selected cluster {selected_cluster} with {len(selected_indices)} gradients")

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot all gradients
    plt.scatter(benign_2d[:, 0], benign_2d[:, 1], c='blue', alpha=0.7, s=50, label='Benign clients')
    plt.scatter(malicious_2d[:, 0], malicious_2d[:, 1], c='red', alpha=0.7, s=50, label='Malicious clients')

    # Draw gray circle around selected cluster
    selected_points = grads_2d[selected_indices]
    center = selected_points.mean(axis=0)
    radius = np.max(np.linalg.norm(selected_points - center, axis=1)) * 1.1

    circle = plt.Circle(center, radius, fill=False, color='gray', linewidth=2,
                       linestyle='--', alpha=0.8, label='Selected for aggregation')
    plt.gca().add_patch(circle)

    # Plot selected gradient (aggregated model) as a star
    selected_2d = pca.transform(selected_grad.reshape(1, -1))
    plt.scatter(selected_2d[0, 0], selected_2d[0, 1], c='black', marker='*', s=200,
               label='Aggregated model', edgecolors='white', linewidth=2)

    # Adjust axis limits to focus on gradients AND the selection circle
    # (ignoring potential outliers in aggregated model if it's too far)
    grad_x_min, grad_x_max = grads_2d[:, 0].min(), grads_2d[:, 0].max()
    grad_y_min, grad_y_max = grads_2d[:, 1].min(), grads_2d[:, 1].max()
    
    # Circle bounds
    circle_x_min, circle_x_max = center[0] - radius, center[0] + radius
    circle_y_min, circle_y_max = center[1] - radius, center[1] + radius
    
    # Combine bounds
    x_min = min(grad_x_min, circle_x_min)
    x_max = max(grad_x_max, circle_x_max)
    y_min = min(grad_y_min, circle_y_min)
    y_max = max(grad_y_max, circle_y_max)
    
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add text annotation
    plt.text(0.02, 0.98, f'Gradients in gray circle aggregated for global model\nMalicious: {len(malicious_grads)}, Benign: {len(benign_grads)}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    json_file = "Result/gradient_analysis/gradients_DPFL_mnist_circle.json"
    output_file = "Result/gradient_analysis/DPFL_gradient_mixed.png"


    if os.path.exists(json_file):
        visualize_gradients_2d(json_file, output_file)
    else:
        print(f"Gradient file not found: {json_file}")
        print("Please run training first to generate gradient data.")