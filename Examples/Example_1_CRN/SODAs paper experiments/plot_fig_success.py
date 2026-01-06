import pandas as pd
import matplotlib.pyplot as plt

# Load the unprocessed results CSV file
results_df = pd.read_csv("unprocessed_fig2_CRN3.csv")

# Specify the noise level to plot
selected_noise = 10.  # Change this to the desired noise level

# Filter the DataFrame for the selected noise level
filtered_df = results_df[results_df["Noise"] == selected_noise]

# Create a binary plot
plt.figure(figsize=(10, 6))

# Plot each trial
for trial in filtered_df["Trial"].unique():
    trial_data = filtered_df[filtered_df["Trial"] == trial]
    plt.plot(
        trial_data["Frequency"],
        trial_data["Success"],
        marker="o",
        linestyle="-",
        label=f"Trial {trial}"
    )

# Add labels, title, and legend
plt.xlabel("Data per initial condition")
plt.ylabel("SODAs success?")
plt.title(f"Successful recovery vs. sampling for {selected_noise}% noise")
plt.yticks([0, 1], labels=["False", "True"])
plt.grid(True)
plt.legend()

# Save and show the plot
output_file = f"figs/noise_figs/full_binary_plot_noise_{selected_noise}.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"Binary plot saved to {output_file}")
