import csv
import os
import statistics

import matplotlib.pyplot as plt


INPUT_FILE = "results/part_e_thermal_log.csv"
FIGURE_FILE = "figures/part_e_clock_temperature.png"
SUMMARY_FILE = "logs/part_e_summary.txt"


rows = []

with open(INPUT_FILE, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows.append(
            {
                "time_seconds": float(row["elapsed_seconds"]),
                "gpu_clock_mhz": float(row["gpu_clock_mhz"]),
                "temperature_c": float(row["temperature_c"]),
                "power_w": float(row["power_w"]),
                "throughput_tflops": float(
                    row["throughput_tflops"]
                ),
                "gpu_utilization_percent": float(
                    row["gpu_utilization_percent"]
                ),
            }
        )


time_minutes = [
    row["time_seconds"] / 60
    for row in rows
]

gpu_clock = [
    row["gpu_clock_mhz"]
    for row in rows
]

temperature = [
    row["temperature_c"]
    for row in rows
]

first_30_seconds = [
    row
    for row in rows
    if row["time_seconds"] <= 30
]

final_five_minutes = [
    row
    for row in rows
    if row["time_seconds"]
    >= rows[-1]["time_seconds"] - 300
]

peak_throughput = max(
    row["throughput_tflops"]
    for row in first_30_seconds
)

steady_state_throughput = statistics.mean(
    row["throughput_tflops"]
    for row in final_five_minutes
)

steady_state_percentage = (
    steady_state_throughput
    / peak_throughput
    * 100
)

throughput_drop_percentage = (
    100 - steady_state_percentage
)

maximum_temperature = max(temperature)
maximum_power = max(
    row["power_w"]
    for row in rows
)

average_utilization = statistics.mean(
    row["gpu_utilization_percent"]
    for row in rows
)


os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Create the required clock and temperature figure.
figure, temperature_axis = plt.subplots(
    figsize=(12, 6)
)

temperature_axis.plot(
    time_minutes,
    temperature,
    color="red",
    linewidth=2,
    label="GPU temperature",
)

temperature_axis.set_xlabel(
    "Time (minutes)"
)

temperature_axis.set_ylabel(
    "Temperature (°C)",
    color="red",
)

temperature_axis.tick_params(
    axis="y",
    labelcolor="red",
)

temperature_axis.grid(
    True,
    linestyle="--",
    alpha=0.4,
)

clock_axis = temperature_axis.twinx()

clock_axis.plot(
    time_minutes,
    gpu_clock,
    color="blue",
    linewidth=2,
    label="GPU clock",
)

clock_axis.set_ylabel(
    "GPU clock (MHz)",
    color="blue",
)

clock_axis.tick_params(
    axis="y",
    labelcolor="blue",
)

plt.title(
    "RTX 3060 Laptop GPU Sustained Load: "
    "Clock and Temperature"
)

figure.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# Save the numerical summary.
with open(SUMMARY_FILE, "w") as file:
    file.write("Part E Sustained Load Summary\n")
    file.write("============================\n\n")
    file.write(
        f"Peak throughput in first 30 seconds: "
        f"{peak_throughput:.4f} TFLOPS\n"
    )
    file.write(
        f"Steady-state throughput in final 5 minutes: "
        f"{steady_state_throughput:.4f} TFLOPS\n"
    )
    file.write(
        f"Steady-state as percentage of peak: "
        f"{steady_state_percentage:.2f}%\n"
    )
    file.write(
        f"Throughput decrease from peak: "
        f"{throughput_drop_percentage:.2f}%\n"
    )
    file.write(
        f"Maximum temperature: "
        f"{maximum_temperature:.1f} C\n"
    )
    file.write(
        f"Maximum power draw: "
        f"{maximum_power:.2f} W\n"
    )
    file.write(
        f"Average GPU utilization: "
        f"{average_utilization:.2f}%\n"
    )
    file.write(
        "Interpretation: The GPU reached its approximately "
        "80 W power ceiling. Temperature-based throttling was "
        "not clearly observed because the maximum temperature "
        "was approximately 77 C.\n"
    )


print("Plot saved to:", FIGURE_FILE)
print("Summary saved to:", SUMMARY_FILE)
print()
print(f"Peak throughput: {peak_throughput:.4f} TFLOPS")
print(
    "Final five-minute throughput: "
    f"{steady_state_throughput:.4f} TFLOPS"
)
print(
    "Steady-state percentage: "
    f"{steady_state_percentage:.2f}%"
)