import json
from collections import Counter
import re
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_PATH = "raw_data/base_raw/relevant_cases_breakdown.json"
OUTPUT_PATH = "cases_per_year.png"

def extract_year(title):
    m = re.search(r"(19|20)\d{2}", title)
    return int(m.group(0)) if m else None

def load_cases(path):
    with open(path, "r") as f:
        return json.load(f)

def count_by_year(cases):
    c = Counter()
    for x in cases:
        y = extract_year(x.get("case_id", ""))
        if y:
            c[y] += 1
    return c

def plot_hist(counts):
    years = sorted(counts.keys())
    values = [counts[y] for y in years]
    plt.figure()
    plt.bar(years, values)
    plt.xlabel("Year")
    plt.ylabel("Number of cases")
    plt.xticks(rotation=45, ha="right")
    plt.title("Cases per year")
    plt.xticks(years, [str(y) for y in years])
    plt.savefig(OUTPUT_PATH)
    plt.close()

def main():
    cases = load_cases(INPUT_PATH)
    counts = count_by_year(cases)
    plot_hist(counts)

if __name__ == "__main__":
    main()
