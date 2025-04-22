import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import os

# Raw data provided by the user (Combined RTX 3080 and RTX 4090)
data_text = """
ON RTX 3080

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 1.5
Total time for this scale (incl. warm-up): 862.89 seconds
Average processing time per image (after 3 warm-ups): 1.478 seconds
Average throughput: 0.68 images/second

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 2.0
Total time for this scale (incl. warm-up): 875.49 seconds
Average processing time per image (after 3 warm-ups): 1.501 seconds
Average throughput: 0.67 images/second

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 2.5
Total time for this scale (incl. warm-up): 945.19 seconds
Average processing time per image (after 3 warm-ups): 1.623 seconds
Average throughput: 0.62 images/second

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 3.0
Total time for this scale (incl. warm-up): 999.80 seconds
Average processing time per image (after 3 warm-ups): 1.723 seconds
Average throughput: 0.58 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 1.5
Total time for this scale (incl. warm-up): 3452.15 seconds
Average processing time per image (after 3 warm-ups): 6.007 seconds
Average throughput: 0.17 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 2.0
Total time for this scale (incl. warm-up): 3466.73 seconds
Average processing time per image (after 3 warm-ups): 6.032 seconds
Average throughput: 0.17 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 2.5
Total time for this scale (incl. warm-up): 3526.76 seconds
Average processing time per image (after 3 warm-ups): 6.135 seconds
Average throughput: 0.16 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 3.0
Total time for this scale (incl. warm-up): 3607.61 seconds
Average processing time per image (after 3 warm-ups): 6.274 seconds
Average throughput: 0.16 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 1.5
Total time for this scale (incl. warm-up): 1786.67 seconds
Average processing time per image (after 3 warm-ups): 3.102 seconds
Average throughput: 0.32 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 2.0
Total time for this scale (incl. warm-up): 1823.24 seconds
Average processing time per image (after 3 warm-ups): 3.166 seconds
Average throughput: 0.32 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 2.5
Total time for this scale (incl. warm-up): 1970.74 seconds
Average processing time per image (after 3 warm-ups): 3.420 seconds
Average throughput: 0.29 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 3.0
Total time for this scale (incl. warm-up): 2114.09 seconds
Average processing time per image (after 3 warm-ups): 3.668 seconds
Average throughput: 0.27 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 1.5
Total time for this scale (incl. warm-up): 4334.93 seconds
Average processing time per image (after 3 warm-ups): 7.546 seconds
Average throughput: 0.13 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 2.0
Total time for this scale (incl. warm-up): 4361.79 seconds
Average processing time per image (after 3 warm-ups): 7.591 seconds
Average throughput: 0.13 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 2.5
Total time for this scale (incl. warm-up): 4508.75 seconds
Average processing time per image (after 3 warm-ups): 7.845 seconds
Average throughput: 0.13 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 3.0
Total time for this scale (incl. warm-up): 4676.93 seconds
Average processing time per image (after 3 warm-ups): 8.135 seconds
Average throughput: 0.12 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 1.5
Total time for this scale (incl. warm-up): 4943.41 seconds
Average processing time per image (after 3 warm-ups): 8.601 seconds
Average throughput: 0.12 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 2.0
Total time for this scale (incl. warm-up): 4973.47 seconds
Average processing time per image (after 3 warm-ups): 8.657 seconds
Average throughput: 0.12 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 2.5
Total time for this scale (incl. warm-up): 5092.03 seconds
Average processing time per image (after 3 warm-ups): 8.861 seconds
Average throughput: 0.11 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 3.0
Total time for this scale (incl. warm-up): 5218.60 seconds
Average processing time per image (after 3 warm-ups): 9.079 seconds
Average throughput: 0.11 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 1.5
Total time for this scale (incl. warm-up): 7471.13 seconds
Average processing time per image (after 3 warm-ups): 13.010 seconds
Average throughput: 0.08 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 2.0
Total time for this scale (incl. warm-up): 7505.87 seconds
Average processing time per image (after 3 warm-ups): 13.069 seconds
Average throughput: 0.08 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 2.5
Total time for this scale (incl. warm-up): 7620.64 seconds
Average processing time per image (after 3 warm-ups): 13.267 seconds
Average throughput: 0.08 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 3.0
Total time for this scale (incl. warm-up): 7770.04 seconds
Average processing time per image (after 3 warm-ups): 13.525 seconds
Average throughput: 0.07 images/second

On RTX4090

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 1.5
Total time for this scale (incl. warm-up): 346.69 seconds
Average processing time per image (after 3 warm-ups): 0.598 seconds
Average throughput: 1.67 images/second

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 2.0
Total time for this scale (incl. warm-up): 360.21 seconds
Average processing time per image (after 3 warm-ups): 0.622 seconds
Average throughput: 1.61 images/second

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 2.5
Total time for this scale (incl. warm-up): 398.50 seconds
Average processing time per image (after 3 warm-ups): 0.688 seconds
Average throughput: 1.45 images/second

Finished processing for Model: prs-eth-thera-edsr-air, Scale: 3.0
Total time for this scale (incl. warm-up): 431.38 seconds
Average processing time per image (after 3 warm-ups): 0.744 seconds
Average throughput: 1.34 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 1.5
Total time for this scale (incl. warm-up): 1694.95 seconds
Average processing time per image (after 3 warm-ups): 2.949 seconds
Average throughput: 0.34 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 2.0
Total time for this scale (incl. warm-up): 1706.41 seconds
Average processing time per image (after 3 warm-ups): 2.969 seconds
Average throughput: 0.34 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 2.5
Total time for this scale (incl. warm-up): 1734.15 seconds
Average processing time per image (after 3 warm-ups): 3.016 seconds
Average throughput: 0.33 images/second

Finished processing for Model: prs-eth-thera-rdn-air, Scale: 3.0
Total time for this scale (incl. warm-up): 1775.08 seconds
Average processing time per image (after 3 warm-ups): 3.086 seconds
Average throughput: 0.32 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 1.5
Total time for this scale (incl. warm-up): 798.68 seconds
Average processing time per image (after 3 warm-ups): 1.387 seconds
Average throughput: 0.72 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 2.0
Total time for this scale (incl. warm-up): 823.03 seconds
Average processing time per image (after 3 warm-ups): 1.430 seconds
Average throughput: 0.70 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 2.5
Total time for this scale (incl. warm-up): 908.01 seconds
Average processing time per image (after 3 warm-ups): 1.577 seconds
Average throughput: 0.63 images/second

Finished processing for Model: prs-eth-thera-edsr-plus, Scale: 3.0
Total time for this scale (incl. warm-up): 992.60 seconds
Average processing time per image (after 3 warm-ups): 1.723 seconds
Average throughput: 0.58 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 1.5
Total time for this scale (incl. warm-up): 2145.04 seconds
Average processing time per image (after 3 warm-ups): 3.734 seconds
Average throughput: 0.27 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 2.0
Total time for this scale (incl. warm-up): 2168.83 seconds
Average processing time per image (after 3 warm-ups): 3.775 seconds
Average throughput: 0.26 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 2.5
Total time for this scale (incl. warm-up): 2254.31 seconds
Average processing time per image (after 3 warm-ups): 3.923 seconds
Average throughput: 0.25 images/second

Finished processing for Model: prs-eth-thera-rdn-plus, Scale: 3.0
Total time for this scale (incl. warm-up): 2346.65 seconds
Average processing time per image (after 3 warm-ups): 4.083 seconds
Average throughput: 0.24 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 1.5
Total time for this scale (incl. warm-up): 2217.60 seconds
Average processing time per image (after 3 warm-ups): 3.857 seconds
Average throughput: 0.26 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 2.0
Total time for this scale (incl. warm-up): 2236.49 seconds
Average processing time per image (after 3 warm-ups): 3.893 seconds
Average throughput: 0.26 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 2.5
Total time for this scale (incl. warm-up): 2311.07 seconds
Average processing time per image (after 3 warm-ups): 4.022 seconds
Average throughput: 0.25 images/second

Finished processing for Model: prs-eth-thera-edsr-pro, Scale: 3.0
Total time for this scale (incl. warm-up): 2379.78 seconds
Average processing time per image (after 3 warm-ups): 4.140 seconds
Average throughput: 0.24 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 1.5
Total time for this scale (incl. warm-up): 3502.11 seconds
Average processing time per image (after 3 warm-ups): 6.098 seconds
Average throughput: 0.16 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 2.0
Total time for this scale (incl. warm-up): 3524.50 seconds
Average processing time per image (after 3 warm-ups): 6.137 seconds
Average throughput: 0.16 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 2.5
Total time for this scale (incl. warm-up): 3600.52 seconds
Average processing time per image (after 3 warm-ups): 6.268 seconds
Average throughput: 0.16 images/second

Finished processing for Model: prs-eth-thera-rdn-pro, Scale: 3.0
Total time for this scale (incl. warm-up): 3681.42 seconds
Average processing time per image (after 3 warm-ups): 6.408 seconds
Average throughput: 0.16 images/second
"""

# Regex patterns
gpu_pattern = re.compile(r"On RTX ?(\d+)", re.IGNORECASE) # Matches "ON RTX 3080" or "On RTX4090"
model_pattern = re.compile(r"Model: (prs-eth-thera-\w+-\w+), Scale: (\d\.\d)")
time_pattern = re.compile(r"Average processing time per image \(after \d+ warm-ups\): (\d+\.\d+) seconds")
throughput_pattern = re.compile(r"Average throughput: (\d+\.\d+) images/second")

# Split data into lines and process
lines = data_text.strip().split('\n')
results = []
current_gpu = None
current_block = []

for line in lines:
    line = line.strip()
    if not line: # Skip empty lines
        # Process the completed block before resetting
        if current_block and "Finished processing" in current_block[0] and current_gpu:
             block_text = "\n".join(current_block)
             model_match = model_pattern.search(block_text)
             time_match = time_pattern.search(block_text)
             throughput_match = throughput_pattern.search(block_text)

             if model_match and time_match and throughput_match:
                 model_name = model_match.group(1)
                 short_model_name = model_name.replace("prs-eth-thera-", "").upper()
                 scale = float(model_match.group(2))
                 avg_time = float(time_match.group(1))
                 throughput = float(throughput_match.group(1))

                 results.append({
                     "GPU": current_gpu, # Add GPU info
                     "Model": short_model_name,
                     "Scale": scale,
                     "Avg Time (s)": avg_time,
                     "Throughput (img/s)": throughput
                 })
        current_block = [] # Reset block
        continue

    gpu_match = gpu_pattern.match(line)
    if gpu_match:
        current_gpu = f"RTX {gpu_match.group(1)}" # Set current GPU (e.g., "RTX 3080")
        # Process the last block of the previous GPU if any
        if current_block and "Finished processing" in current_block[0] and current_gpu: # Need old current_gpu here
             # This logic might be slightly off, better to process blocks when they end (empty line)
             pass # Handled by empty line logic now
        current_block = [] # Reset block for new GPU section start
    else:
        current_block.append(line) # Add line to current block

# Process the very last block after the loop finishes
if current_block and "Finished processing" in current_block[0] and current_gpu:
     block_text = "\n".join(current_block)
     model_match = model_pattern.search(block_text)
     time_match = time_pattern.search(block_text)
     throughput_match = throughput_pattern.search(block_text)

     if model_match and time_match and throughput_match:
         model_name = model_match.group(1)
         short_model_name = model_name.replace("prs-eth-thera-", "").upper()
         scale = float(model_match.group(2))
         avg_time = float(time_match.group(1))
         throughput = float(throughput_match.group(1))

         results.append({
             "GPU": current_gpu, # Add GPU info
             "Model": short_model_name,
             "Scale": scale,
             "Avg Time (s)": avg_time,
             "Throughput (img/s)": throughput
         })


# Create DataFrame
df = pd.DataFrame(results)

# --- Plotting ---
sns.set_theme(style="whitegrid")

# Define output directory
output_dir = "inference_plots_comparison"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Plot 1: Average Processing Time vs. Scale (Comparing GPUs) ---
# Use relplot to create facets for each GPU
time_plot = sns.relplot(
    data=df,
    x="Scale",
    y="Avg Time (s)",
    hue="Model",
    col="GPU",  # Create separate columns for each GPU
    kind="line",
    marker="o",
    linewidth=2.5,
    facet_kws={'sharey': False} # Don't share Y-axis, as scales might differ
)
time_plot.fig.suptitle('Average Processing Time per Image vs. Scale Factor', y=1.03) # Add overall title
time_plot.set_axis_labels("Scale Factor", "Average Time per Image (s)")
time_plot.set_titles("GPU: {col_name}") # Set titles for each subplot
time_plot.set(xticks=df['Scale'].unique()) # Ensure all scale factors are marked on x-axis

# Save the figure
time_filename = os.path.join(output_dir, "processing_time_vs_scale_gpu_comparison.png")
plt.savefig(time_filename, dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free memory
print(f"Saved processing time comparison plot to: {time_filename}")


# --- Plot 2: Average Throughput vs. Scale (Comparing GPUs) ---
throughput_plot = sns.relplot(
    data=df,
    x="Scale",
    y="Throughput (img/s)",
    hue="Model",
    col="GPU", # Create separate columns for each GPU
    kind="line",
    marker="o",
    linewidth=2.5,
    facet_kws={'sharey': False} # Don't share Y-axis
)
throughput_plot.fig.suptitle('Average Throughput vs. Scale Factor', y=1.03) # Add overall title
throughput_plot.set_axis_labels("Scale Factor", "Average Throughput (images/second)")
throughput_plot.set_titles("GPU: {col_name}") # Set titles for each subplot
throughput_plot.set(xticks=df['Scale'].unique()) # Ensure all scale factors are marked on x-axis

# Save the figure
throughput_filename = os.path.join(output_dir, "throughput_vs_scale_gpu_comparison.png")
plt.savefig(throughput_filename, dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free memory
print(f"Saved throughput comparison plot to: {throughput_filename}")


# --- Print Data Summary Table ---
print("\nData Summary Table:")
# Pivot to show GPUs as columns and models/scales as rows/columns
summary_table = df.pivot_table(
    index=['Model', 'Scale'],
    columns='GPU',
    values=['Avg Time (s)', 'Throughput (img/s)']
)
print(summary_table)

# Optional: Save summary table to CSV
summary_filename = os.path.join(output_dir, "inference_summary.csv")
summary_table.to_csv(summary_filename)
print(f"\nSaved summary table to: {summary_filename}")