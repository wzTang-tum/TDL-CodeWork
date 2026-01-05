import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Data
methods = ['AttackFL', 'AttackNaive', 'AttackNonDP', 'AttackDPFL']
# Extracted from the last entry of each JSON file
accuracy = [55.12, 50.12, 60.12, 32.12]
asr = [100.00, 20.12, 25.12, 99.12]

# Settings
x = np.arange(2)
bar_width = 0.14       # Width of each bar
group_gap = 0.03       # Gap between bars within a group
step = bar_width + group_gap
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * step

# Lighter, pastel colors
colors = ['#85C1E9', '#F8C471', '#82E0AA', '#F1948A']

fig, ax = plt.subplots(figsize=(10, 6))

# Styling: Clean look
ax.set_facecolor('white')
ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#AAAAAA')
ax.tick_params(axis='both', which='both', length=0) # Hide ticks

# Legend handles
legend_handles = []

# Plotting
for i, method in enumerate(methods):
    values = [accuracy[i], asr[i]]
    xs = x + offsets[i]
    
    # Add invisible bar to ensure correct axis scaling and layout
    ax.bar(xs, values, bar_width, color=colors[i], label=method, alpha=0)
    
    # Create handle for legend
    legend_handles.append(mpatches.Patch(color=colors[i], label=method))
    
    for j, val in enumerate(values):
        # Draw rounded rectangle
        # boxstyle="round" gives rounded corners. 
        # rounding_size controls the radius.
        patch = mpatches.FancyBboxPatch(
            (xs[j] - bar_width/2, 0), bar_width, val,
            boxstyle="round,pad=0,rounding_size=0.03",
            ec=None, fc=colors[i], linewidth=0, # No border
            mutation_aspect=1.0,
            zorder=3
        )
        ax.add_patch(patch)
        
        # Add value label
        ax.text(xs[j], val + 1.5, f'{val:.1f}', ha='center', va='bottom', 
                fontsize=9, color='#444444', fontweight='bold', zorder=4)

# Labels and Title
ax.set_ylabel('scores', fontsize=11, color='#555555')
ax.set_title('Performance Comparison on FEMNIST', fontsize=14, pad=20, color='#333333', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'ASR'], fontsize=12, color='#333333')
ax.set_ylim(0, 115) # Extra space for labels

# Legend: Centered at the top
ax.legend(handles=legend_handles, frameon=False, loc='upper center', ncol=4, fontsize=10)

plt.tight_layout()
output_path = 'd:/TUM/TDL/Robust-DPFL-main/Result/femnist_bar_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {output_path}")
