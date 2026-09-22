import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Read all naive-attention CSV files
csv_files = glob.glob("results/part_d_naive*.csv")

if not csv_files:
    raise FileNotFoundError("No Part D naive-attention CSV files found.")

dataframes = []

for file in csv_files:
    dataframe = pd.read_csv(file)
    dataframes.append(dataframe)

data = pd.concat(dataframes, ignore_index=True)

# Convert columns to numeric
data["sequence_length"] = pd.to_numeric(
    data["sequence_length"],
    errors="coerce",
)

data["peak_memory_gb"] = pd.to_numeric(
    data["peak_memory_gb"],
    errors="coerce",
)

# Prefer successful results when duplicate sequence lengths exist
data["success_flag"] = data["status"].eq("success")

data = (
    data.sort_values(["sequence_length", "success_flag"])
    .drop_duplicates("sequence_length", keep="last")
    .sort_values("sequence_length")
)

successful = data[
    (data["status"] == "success")
    & data["peak_memory_gb"].notna()
].copy()

failed = data[data["status"] == "out_of_memory"].copy()

if len(successful) < 3:
    raise RuntimeError("At least three successful measurements are needed.")

x = successful["sequence_length"].to_numpy(dtype=float)
y = successful["peak_memory_gb"].to_numpy(dtype=float)

# Fit:
# peak_memory = intercept + coefficient * sequence_length^2
x_squared = x ** 2

coefficient, intercept = np.polyfit(x_squared, y, 1)

predicted = intercept + coefficient * x_squared

ss_residual = np.sum((y - predicted) ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)
r_squared = 1.0 - (ss_residual / ss_total)

# Convert coefficient from GB/token^2 to bytes/token^2
coefficient_bytes = coefficient * (1024 ** 3)

# Smooth quadratic curve for the plot
curve_x = np.linspace(x.min(), x.max(), 300)
curve_y = intercept + coefficient * (curve_x ** 2)

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    s=70,
    label="Measured peak memory",
)

plt.plot(
    curve_x,
    curve_y,
    linewidth=2,
    label="Quadratic fit",
)

plt.xlabel("Sequence length")
plt.ylabel("Peak allocated memory (GB)")
plt.title("Naive Attention Memory Growth")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

figure_file = "figures/part_d_naive_peak_memory.png"
plt.savefig(figure_file, dpi=200)
plt.close()

# Determine tested boundary
largest_success = int(successful["sequence_length"].max())

if len(failed) > 0:
    smallest_failure = int(failed["sequence_length"].min())
else:
    smallest_failure = "Not found"

# Save fitting details
fit_file = "results/part_d_quadratic_fit.txt"

with open(fit_file, "w") as file:
    file.write("Part D Naive Attention Quadratic Memory Fit\n")
    file.write("============================================\n")
    file.write(f"Number of successful measurements: {len(successful)}\n")
    file.write(f"Quadratic coefficient: {coefficient:.12e} GB/token^2\n")
    file.write(f"Quadratic coefficient: {coefficient_bytes:.6f} bytes/token^2\n")
    file.write(f"Intercept: {intercept:.6f} GB\n")
    file.write(f"R-squared: {r_squared:.6f}\n")
    file.write(f"Largest tested success: {largest_success}\n")
    file.write(f"Smallest tested failure: {smallest_failure}\n")

print("Successful measurements used for fitting:")
print(successful[["sequence_length", "peak_memory_gb"]].to_string(index=False))

print("\nLargest tested success:", largest_success)
print("Smallest tested failure:", smallest_failure)

print("\nQuadratic coefficient:")
print(f"{coefficient:.12e} GB/token^2")

print("\nQuadratic coefficient in bytes:")
print(f"{coefficient_bytes:.6f} bytes/token^2")

print("\nR-squared:")
print(f"{r_squared:.6f}")

print("\nFigure saved to:", figure_file)
print("Fit details saved to:", fit_file)

