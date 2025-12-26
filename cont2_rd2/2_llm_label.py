import json
import os
from openai import OpenAI

CASE_TYPES = {"privacy", "ip", "antitrust", "justice", "tort", "consumer"}

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
                "core_ai_system": {"type": "string"},
                "plaintiff_labels": {
                    "type": "array", 
                    "items": {
                        "type": "string"
                    }
                }, 
                "plaintiff_name": {"type": "string"},
                "defendant_labels": {
                    "type": "array", 
                    "items": {
                        "type": "string"
                    }
                }, 
                "defendant_name": {"type": "string"},
            },
            "required": ["case_id", "core_ai_system", "plaintiff_labels", "plaintiff_name", "defendant_labels", "defendant_name"]
        }
    }

    LABEL_DESCRIPTIONS = """
        1. Antitrust: market competition, monopolization, market power, anti-competitive practices, market dominance, price-fixing, exclusive dealing, or restraint of trade involving ANY tech companies, or anti-competitive practices by major platforms or AI companies.
        2. IP Law: patents, copyrights, trademarks for AI models or tech, or training data disputes, AI-generated content ownership.
        3. Privacy and Data Protection: data breaches, unauthorized data collection by automated systems, or privacy violations involving algorithms or data processing.
        4. Tort: physical harm, emotional distress, negligence, defamation, or personal injury involving ANY automated systems, tech systems, major tech corporations using AI, or algorithms.
        5. Justice and Equity: discrimination, bias, civil rights violations, equal protection issues, equity issues, fairness concerns, or systemic bias, alleged or substantiated discrimination or bias caused by AI, automated systems, or algorithms, or related to AI, automated systems, or algorithims. (e.g., hiring, lending, search).
        6. Consumer Protection: deceptive practices, unfair business practices with tech/automated systems, or misleading marketing of tech products or AI capabilities.
        7. AI in Legal Proceedings: AI systems are merely used in the court processes, legal case management, or litigation tools. The core contention is not about AI, but AI tools have been used in the litigation process.
        8. Unrelated: cases that have no meaningful connection to AI, ML, or automated systems. If the case is frivolous, or involves discrimination, privacy, or other issues **without automation/AI/algorithmic involvement**, classify as Unrelated.
        """

    prompt = f"""
        Extract information from this court case.

        CASE NAME: {case_data['name']}
        SUMMARY: {case_data['summary']}

        LABEL DESCRIPTIONS: {LABEL_DESCRIPTIONS}
        
        Follow instructions:
        - Fill in the JSON fields only
        - For case_id, use the case name/filename
        - For core_ai_system: Read through the entire case text, extract the text describing where AI is mentioned, and the AI technology used.
        - Given the label descriptions, write the label into the JSON if the litigant used an argument that fits the description of the label.
        - Be specific about outcomes
        - For plaintiff_name / defendant_name: Describe the party.
        """
    
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
    for case_type in CASE_TYPES:
        process_cases(f"{case_type}/{case_type}.json", f"{case_type}/cases_breakdown.json")