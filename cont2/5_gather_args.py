import json


TOPIC = "tort"

INPUT_PATH = f"{TOPIC}/cases_breakdown.json"
P_OUTPUT_PATH = f"{TOPIC}/plaintiff_args.txt"
D_OUTPUT_PATH = f"{TOPIC}/defendant_args.txt"

# Passing JSON -> txt to NotebookLM
prompt_defense = f"""
Look through every defense and legal_basis in the dataset. 
Identify the main argument categories that appear across the cases.
Be specific in describing each category, and ensure the categories reflect all arguments in the dataset.
Expected output should be a list of categories, their arguments, and the legal basis for each argument. 
""" 

prompt_plaintiff = f"""
Look through every claim and legal_basis in the dataset. 
Identify the main argument categories that appear across the cases. 
Be specific in describing each category, and ensure the categories reflect all arguments in the dataset.
Expected output should be a list of categories, their arguments, and the legal basis for each argument. 
"""


def get_args(cases):
    plaintiff_args = []
    defendant_args = []

    for case in cases:
        claims = case.get("claims", [])
        defenses = case.get("defenses", [])

        for claim in claims:
            to_append = {}
            to_append["claim"] = claim.get("claim", "")
            to_append["legal_basis"] = claim.get("legal_basis", "")
            plaintiff_args.append(to_append)
        for defense in defenses:
            defendant_args.append(defense)

    all_p = {
        "plaintiff_args": plaintiff_args,
    }

    all_d = {
        "defendant_args": defendant_args
    }

    return all_p, all_d

def main():

    # Load cases
    with open(INPUT_PATH, "r") as f:
        cases = json.load(f)

    print(f"✓ Loaded {len(cases)} cases")

    all_p, all_d = get_args(cases)

    # Save output
    with open(P_OUTPUT_PATH, 'w') as file:
        file.write(json.dumps(all_p, indent=2))

    with open(D_OUTPUT_PATH, 'w') as file:
        file.write(json.dumps(all_d, indent=2))

    print(f"✓ Completed successfully.")


if __name__ == "__main__":
    main()