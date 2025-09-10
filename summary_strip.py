# try and jsut get the summaries from the json
import json
INPUT_FILE_PATH = "categorized_cases.json"
OUTPUT_FILE_PATH = "summaries.json"

def extract_summaries(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    summaries = [case_info["summary"] for case_info in data.values() if "summary" in case_info]
    return summaries

if __name__ == "__main__":
    summaries = extract_summaries(INPUT_FILE_PATH)
    with open(OUTPUT_FILE_PATH, "w") as json_file:
        json.dump(summaries, json_file, indent=4)
