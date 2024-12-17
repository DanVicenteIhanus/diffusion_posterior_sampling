"""
Call: poetry run python scripts/results_to_latex.py path_to_txt_file

Description: prints latex code for a table of the results in the txt file.

Note: results are stored in the output folder as txt files *_metrics_log.txt

Example: poetry run python scripts/results_to_latex.py output/celeba_metrics_log.txt
"""

import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Parse metrics from a text file and output as LaTeX table.")
parser.add_argument("file", type=str, help="Path to the input text file containing metrics.")
args = parser.parse_args()

metrics = {}

with open(args.file, "r") as file:
    lines = file.readlines()

current_measurement = None

for line in lines:
    line = line.strip()
    if line.startswith("Measurement:"):
        current_measurement = line.split(":")[1].strip()
        if current_measurement: 
            metrics[current_measurement] = {}
    elif current_measurement:
        match = re.match(r"(\w+):\s+([\d.]+)(\s\+\-/\s[\d.]+)?", line)
        if match:
            key = match.group(1)
            value = float(match.group(2))
            metrics[current_measurement][key] = value

df = pd.DataFrame(metrics)
print(df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format))
