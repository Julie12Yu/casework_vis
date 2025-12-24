import json

UNRELATED = "Unrelated"
AILEGAL = "AI in Legal Proceedings"

def filter_privacy_cases(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cases = data.get('documents', [])

    filtered_cases = []
    for case in cases:
        if isinstance(case, dict):
            if case.get('legal_category_name') != UNRELATED and case.get('legal_category_name') != AILEGAL :
                filtered_cases.append(case)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_cases, f, indent=2, ensure_ascii=False)

    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Specify your input and output file paths
    input_file = "misc/new_court_cases_processed.json"
    output_file = f"all_cases.json"
    
    filter_privacy_cases(input_file, output_file)