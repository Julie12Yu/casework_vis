all prompts used in the process of creating contribution 2

after contribution 1, we had a lot of cases with a label of what category they fell under.
we are investigating specific categories, so while looking at the 2 catgories we care for, we wanted to find arguments for each case.

breakdown.py

- looks at each case in the category provided
- looks at the raw text.
- finds plaintiffs + defendants, generates description for it
- finds arguments for litigants (both sides), finds descriptions for it.
- there are 2 parts: a schema, and a prompt.

```
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
                    "plaintiff_name": {"type": "string"},
                    "plaintiff_description": {"type": "string"},
                    "defendant_name": {"type": "string"},
                    "defendant_description": {"type": "string"}
                },
                "required": ["plaintiff_name", "plaintiff_description", "defendant_type", "defendant_description"]
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
```

```
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
    - For plaintiff_name / defendant_name: Find the name of the party and assign it to this category. In 5 words or less.
    - For plaintiff_description / defendant_description: Describe the party (e.g., “individual”, “corporation”, “business”, “government agency”, “class-action group”). In 10 words or less.
    - For ai_relevance: If artificial intelligence, machine learning, automated systems, algorithmic technologies, or other AI/ML-related technologies are involved in, impact, or led to the dispute, return exactly "NOT RELATED". Otherwise, describe how they play a role."""
```
