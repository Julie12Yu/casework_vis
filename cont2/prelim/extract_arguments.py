import json
from openai import OpenAI

# Initialize client
with open("otherkey.txt") as f:
    key = f.read().strip()
client = OpenAI(api_key=key)

INPUT_PATH = "/Users/julie12yu/development/casework_vis/privacy_summary.txt"
OUTPUT_PATH = "/Users/julie12yu/development/casework_vis/privacy_args_breakdown.json"

def main():
    combined_results = {}

    with open(INPUT_PATH, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(lines, start=1):
        print(f"Processing line {idx}/{len(lines)}...")
        case_entry = {"input": line}

        # --- Plaintiff ---
        try:
            prompt = f"You are a legal expert. Identify the plaintiff's arguments:\n{line}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            case_entry["plaintiff_arg"] = response.choices[0].message.content.strip()
        except Exception as e:
            case_entry["plaintiff_arg"] = f"Error: {str(e)}"

        # --- Defendant ---
        try:
            prompt = f"You are a legal expert. Identify the defendant's arguments:\n{line}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            case_entry["defendant_arg"] = response.choices[0].message.content.strip()
        except Exception as e:
            case_entry["defendant_arg"] = f"Error: {str(e)}"

        # --- Outcome ---
        try:
            prompt = f"You are a legal expert. Identify the case outcome and its importance:\n{line}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            case_entry["result"] = response.choices[0].message.content.strip()
        except Exception as e:
            case_entry["result"] = f"Error: {str(e)}"

        combined_results[f"line_{idx}"] = case_entry

    with open(OUTPUT_PATH, "w") as f:
        json.dump(combined_results, f, indent=2)

    print("All responses saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
