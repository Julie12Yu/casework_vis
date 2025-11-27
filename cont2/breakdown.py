import json
import os
from openai import OpenAI

def extract_case_structure(case_data):
    with open("../otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)

    schema = {
        "name": "case_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "case_name": {"type": "string"},
                "ai_relevance": {"type": "string"},
                "parties": {
                    "type": "object",
                    "properties": {
                        "plaintiff_type": {"type": "string"},
                        "defendant_type": {"type": "string"}
                    },
                    "required": ["plaintiff_type", "defendant_type"]
                },
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "legal_basis": {"type": "string"},
                            "outcome": {"type": "string"}
                        },
                        "required": ["claim", "legal_basis", "outcome"]
                    }
                },
                "defenses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "defense": {"type": "string"},
                            "legal_basis": {"type": "string"}
                        },
                        "required": ["defense", "legal_basis"]
                    }
                }
            },
            "required": ["case_id", "case_name", "ai_relevance", "parties", "claims", "defenses"]
        }
    }

    prompt = f"""
        Extract information from this court case.

        CASE NAME: {case_data['name']}
        SUMMARY: {case_data['summary']}

        Follow instructions:
        - Fill in the JSON fields only
        - For case_id, use the case name/filename
        - Extract ALL claims and defenses
        - Use natural language
        - Be specific about outcomes
        - For plaintiff_type / defendant_type: Briefly describe each party in 5 words or less (e.g., “individual”, “corporation”, “business”, “government agency”, “class-action group”). Do not give names; give only the category-level description.
        - For ai_relevance: If artificial intelligence, machine learning, automated systems, algorithmic technologies, or other AI/ML-related technologies are involved in, impact, or led to the dispute, return exactly "NOT RELATED". Otherwise, describe how they play a role."""
    
    response = client.chat.completions.create(
        model="gpt-5.1",
        response_format={
            "type": "json_schema",
            "json_schema": schema
        },
        messages=[
            {"role": "system", "content": "Return valid JSON ONLY, following the schema."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

def process_cases(input_file, output_file):
    with open(input_file, 'r') as f:
        cases = json.load(f)
    
    results = []
    
    for i, case in enumerate(cases):
        print(f"Processing case {i+1}/{len(cases)}: {case['name']}")
        
        try:
            extracted = extract_case_structure(case)
            print(f"Extracted data: {extracted}")
            results.append(extracted)
        except Exception as e:
            print(f"Error processing case {case['name']}: {e}")
            results.append({
                "case_id": case['name'],
                "error": str(e)
            })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed {len(results)} cases")
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    process_cases("privacy/Privacy and Data Protection.json", "privacy/priv_cases_breakdown.json")