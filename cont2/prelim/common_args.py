#!/usr/bin/env python3
import json
import os
import sys
from openai import OpenAI

INPUT_PATH = "/Users/julie12yu/development/casework_vis/privacy_args_breakdown.json"


PROMPT_TEMPLATE = """You are analyzing multiple legal case snippets about AI/IP.
Each item has an "input" (neutral summary) plus either plaintiff or defendant arguments.

TASK:
1) Identify the common patterns/arguments recurring across these items for the specified SIDE.
2) Use bullet points to express points.
3) Use precise legal/AI terms when helpful (e.g., fair use, market harm, registration, substantial similarity, trade secrets, willfulness).
4) Be concise—no more than ~300 words.

SIDE: {side}

TEXT:
{payload}
"""

def load_json(INPUT_PATH):
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_text(data, side):
    """side is 'plaintiff' or 'defendant'."""
    chunks = []
    for key in sorted(data.keys()):
        item = data[key]
        inp = item.get("input", "")
        if side == "plaintiff":
            arg = item.get("plaintiff_arg", "")
        else:
            arg = item.get("defendant_arg", "")
        if inp or arg:
            chunks.append(f"- INPUT: {inp}\n- ARG: {arg}")
    return "\n\n".join(chunks)

def trim(text, max_chars=18000):
    # keep it simple: trim to avoid overlong prompts
    return text[:max_chars]

def ask_gpt(client, model, side, payload):
    prompt = PROMPT_TEMPLATE.format(side=side.upper(), payload=payload)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a legal expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def main():
    try:
        data = load_json(INPUT_PATH)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        sys.exit(1)

    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)
    model = "gpt-4o"

    # Build payloads
    plaintiff_payload = trim(collect_text(data, "plaintiff"))
    defendant_payload = trim(collect_text(data, "defendant"))

    # Query GPT-4o
    print("\n=== Common Plaintiff Arguments ===\n")
    try:
        print(ask_gpt(client, model, "plaintiff", plaintiff_payload))
    except Exception as e:
        print(f"(Error from GPT for plaintiff side) {e}")

    print("\n=== Common Defendant Arguments ===\n")
    try:
        print(ask_gpt(client, model, "defendant", defendant_payload))
    except Exception as e:
        print(f"(Error from GPT for defendant side) {e}")

if __name__ == "__main__":
    main()