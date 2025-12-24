import json
import pandas as pd
from pandas import json_normalize
import json

UNRELATED = "Unrelated"
AILEGAL = "AI in Legal Proceedings"
OUTPUT = "relevant_cases"

def filter_privacy_cases(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cases = data.get('documents', [])

    filtered_cases = []
    for case in cases:
        if isinstance(case, dict):
            if case.get('legal_category_name') != UNRELATED and case.get('legal_category_name') != AILEGAL :
                filtered_case = {
                    "name": case.get("name"),
                    "summary": case.get("summary"),
                    "legal_category_name": case.get("legal_category_name")
                }
                filtered_cases.append(filtered_case)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_cases, f, indent=2, ensure_ascii=False)

    print(f"Output saved to: {output_file}")

def json_to_csv_pandas(json_file_path, csv_file_path):
    with open(json_file_path, encoding='utf-8') as f:
        data = json.load(f)

    df = json_normalize(data)
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"Successfully converted {json_file_path} to {csv_file_path}")


if __name__ == "__main__":
    input_file = "misc/new_court_cases_processed.json"
    output_json_file = f"{OUTPUT}.json"
    
    filter_privacy_cases(input_file, output_json_file)

    output_csv_file = f"{OUTPUT}.csv"
    json_to_csv_pandas(output_json_file, output_csv_file) 