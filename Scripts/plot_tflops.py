import csv
import matplotlib.pyplot as plt

input_file =  r"C:\Users\samru\Study\Study_Abroad\Universities\SJSU\Semester3\Data_266_Generative_AI\Assignments\GPU_Assignment1\results\matmul_results.csv"
output_file =  r"C:\Users\samru\Study\Study_Abroad\Universities\SJSU\Semester3\Data_266_Generative_AI\Assignments\GPU_Assignment1\figures\achieved_tflops.png"

data = {
    "FP32": {"size": [], "tflops": []},
    "TF32": {"size": [], "tflops": []},
    "FP16": {"size": [], "tflops": []},
    "BF16": {"size": [], "tflops": []},
}

with open(input_file, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["status"] == "success":
            precision = row["precision"]
            data[precision]["size"].append(int(row["matrix_size"]))
            data[precision]["tflops"].append(float(row["achieved_tflops"]))

for precision, values in data.items():
    plt.plot(
        values["size"],
        values["tflops"],
        marker="o",
        linewidth=2,
        label=precision,
    )

plt.xscale("log", base=2)
plt.xlabel("Matrix size N")
plt.ylabel("Achieved TFLOPS")
plt.title("RTX 3060 Laptop GPU: Matrix Multiplication Throughput")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Figure saved to {output_file}")