import csv

input_file = r"C:\Users\samru\Study\Study_Abroad\Universities\SJSU\Semester3\Data_266_Generative_AI\Assignments\GPU_Assignment1\results\matmul_results.csv"

output_file = r"C:\Users\samru\Study\Study_Abroad\Universities\SJSU\Semester3\Data_266_Generative_AI\Assignments\GPU_Assignment1\results\matmul_results_with_percentages.csv"

THEORETICAL_PEAK_TFLOPS = {
    "FP32": 13.08,
    "TF32": 26.16,
    "FP16": 52.32,
    "BF16": 52.32,
}

with open(input_file, newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

for row in rows:
    precision = row["precision"]

    if row["status"] == "success":
        achieved = float(row["achieved_tflops"])
        theoretical = THEORETICAL_PEAK_TFLOPS[precision]
        percentage = achieved / theoretical * 100

        row["theoretical_peak_tflops"] = theoretical
        row["percent_of_theoretical_peak"] = round(percentage, 2)
    else:
        row["theoretical_peak_tflops"] = THEORETICAL_PEAK_TFLOPS[precision]
        row["percent_of_theoretical_peak"] = ""

fieldnames = list(rows[0].keys())

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated results saved to:")
print(output_file)