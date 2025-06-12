import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# List of CSV file paths and their corresponding labels
csv_files = [
    ("processed_fig1_deg2.csv", "p=2, |\Theta| = 14"),
    ("processed_fig1_deg3.csv", "p=3, |\Theta| = 34"),
    ("processed_fig1_deg4.csv", "p=4, |\Theta| = 69"),
]

plt.figure(figsize=(10, 7))

is_first = True # For legend-making
for file_path, label in csv_files:
    df = pd.read_csv(file_path)
    
    # Get noise, stds
    summary_stats = (
        df.groupby("Noise")["Frequency"]
        .agg(["mean", "std", "size"])
        .reset_index()
    )
    print(f'{file_path}\n')
    print('---------\n')
    print(summary_stats)
    print('\n')

    # Get vals to plot
    noise_levels = summary_stats["Noise"]
    means = summary_stats["mean"]
    sizes = summary_stats["size"]
    errors = summary_stats["std"] / np.sqrt(sizes)

    # Locate points where non robust
    flagged = [(i, float(noise_levels[i]), float(means[i])) for i in range(len(means)) if sizes[i] <= 8]

    # Plot w error bars
    if label == "p=2, |\Theta| = 14":
        ls = '--'
    elif label == "p=3, |\Theta| = 34":
        ls = '-'
    else:
        ls = ':'

    
    if len(flagged) != 0:
        inds, flag_x, flag_y = zip(*flagged)
        first_x = inds[0]

        eb = plt.errorbar(noise_levels[:first_x], means[:first_x], yerr=errors[:first_x], fmt='-o', ls=ls, color='#343084',
            elinewidth=.9, capsize=5, lw=.9, ms=6, label=f"${label}$", capthick=0.9)

        if is_first: # Only record legend for first data set
            is_first = False
            plt.plot(list(flag_x), list(flag_y), 'x', ms=8, label='Non-robust recovery', color='#343084',)
            plt.plot(noise_levels, means, ls=ls, ms=0, label=None, color='#343084', alpha=0.25)
        else:
            plt.plot(list(flag_x), list(flag_y), 'x', ms=8, label=None, color='#343084',)
            plt.plot(noise_levels, means, ls=ls, ms=0, label=None, color='#343084', alpha=0.25)
    else:
        eb = plt.errorbar(noise_levels, means, yerr=errors, fmt='-o', ls=ls, color='#343084',
            elinewidth=.9, capsize=5, lw=.9, ms=6, label=f"${label}$", capthick=.9)

# Adjust ticks
plt.xticks(np.linspace(0, 16, 9, True), fontsize=18)
plt.yticks(fontsize=18)

# Add labels, title
plt.xlabel("% Gaussian noise", fontsize=18)
plt.ylabel("Data points per initial condition", fontsize=18)
plt.title("Data required for CRN1 recovery", fontsize=20)
plt.grid(True, linewidth=0.5)

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
order = [3,2,1,0] # Reorder w/ flagged on bottom
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=18)

output_file = "fig_1"
os.makedirs("figs/" + output_file, exist_ok=True)

# Save as svg
plt.savefig(f"figs/{output_file}/{output_file}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"figs/{output_file}/{output_file}.pdf", bbox_inches='tight') 
plt.savefig(f"figs/{output_file}/{output_file}.svg", bbox_inches='tight')

print(f"Plots saved to {output_file} dir")
plt.show()