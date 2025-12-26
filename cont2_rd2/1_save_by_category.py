import json

# Privacy and Data Protection
# IP Law
# Consumer Protection
# Tort
# Justice and Equtiy
NAME = 'Antitrust'
NAME_FILE = 'antitrust'

def filter_privacy_cases(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cases = data.get('documents', [])

    filtered_cases = []
    for case in cases:
        if isinstance(case, dict):
            if case.get('legal_category_name') == NAME:
                filtered_cases.append(case)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_cases, f, indent=2, ensure_ascii=False)

    print(f"Total cases found in category '{NAME}': {len(filtered_cases)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Specify your input and output file paths
    input_file = "../misc/new_court_cases_processed.json"
    output_file = f"labeled_data/{NAME_FILE}/{NAME_FILE}.json"
    
    filter_privacy_cases(input_file, output_file)