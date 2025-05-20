import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# List of CSV file paths and their corresponding labels
csv_files = [
    ("processed_fig2_CRN3.csv", ("CRN3, ", "|\Theta| = 119")),
    ("processed_fig2_CRN2.csv", ("CRN2, ", "|\Theta| = 35")),
    ("processed_fig1_deg2.csv", ("CRN1, ", "|\Theta| = 14")),
]

plt.figure(figsize=(10, 7))

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

    if label[0] == "CRN1, ":
        color = "#343084"
    elif label[0] == "CRN2, ":
        color = "#AA469A"
    else:
        color = '#18793D'

    # Grab points where non robust
    flagged_CRN2 = [(i, float(noise_levels[i]), float(means[i])) for i in range(12, 16)]
    flagged_CRN3 = [(i, float(noise_levels[i]), float(means[i])) for i in range(11, 16)]
    inds_2, flag_x_2, flag_y_2 = zip(*flagged_CRN2)
    inds_3, flag_x_3, flag_y_2 = zip(*flagged_CRN3)
    first_x_2 = inds_2[0]
    first_x_3 = inds_3[0]

    if label[0] == "CRN1, ":
        eb = plt.errorbar(noise_levels, means, yerr=errors, fmt='-o', ls='-', color=color,
            elinewidth=.9, capsize=5, lw=.9, ms=6, label=f"{label[0]}${label[1]}$", capthick=.9)
    elif label[0] == "CRN2, ":
        eb = plt.errorbar(noise_levels[:12], means[:12], yerr=errors[:12], fmt='-o', ls='-', color=color,
            elinewidth=.9, capsize=5, lw=.9, ms=6, label=f"{label[0]}${label[1]}$", capthick=.9)
        
        plt.plot(list(noise_levels[12:]), list(means[12:]), 'x', ms=8, label='Non-robust recovery', color='black',)
        plt.plot(list(noise_levels[12:]), list(means[12:]), 'x', ms=8, label=None, color=color,)
        plt.plot(noise_levels, means, ls='-', ms=0, label=None, color=color, alpha=0.25)

    elif label[0] == "CRN3, ":
        eb = plt.errorbar(noise_levels[:11], means[:11], yerr=errors[:11], fmt='-o', ls='-', color=color,
            elinewidth=.9, capsize=5, lw=.9, ms=6, label=f"{label[0]}${label[1]}$", capthick=.9)
        

        plt.plot(list(noise_levels[11:]), list(means[11:]), 'x', ms=8, label=None, color=color,)
        plt.plot(noise_levels, means, ls='-', ms=0, label=None, color=color, alpha=0.25)

# Adjust ticks
plt.xticks(np.linspace(0,15,16, True), fontsize=18)
plt.yticks(np.linspace(0,1200,7, True), fontsize=18)

# Add labels, title
plt.xlabel("% Gaussian noise", fontsize=18)
plt.ylabel("Data points per initial condition", fontsize=18)
plt.title("Data required for recovery ($p=2$)", fontsize=20)
plt.grid(True, linewidth=0.5)

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,3,0] # Reorder w/ flagged on bottom
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=18)

output_file = "fig_2"
os.makedirs("figs/" + output_file, exist_ok=True)

# Save as svg
plt.savefig(f"figs/{output_file}/{output_file}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"figs/{output_file}/{output_file}.pdf", bbox_inches='tight') 
plt.savefig(f"figs/{output_file}/{output_file}.svg", bbox_inches='tight')

print(f"Plots saved to {output_file} dir")
plt.show()