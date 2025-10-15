import json
from openai import OpenAI
import sys
import time
import re

CLASSIFY_INSTRUCTIONS = '''You are a legal expert specialized in categorizing cases based on the aspect of the case specifically related to AI. 
The case title is: {title}
The summary of the case is: {text}
Reading a summary of the case, then classify the case based on the definition below:

1. Antitrust: refers to cases where the defendant is accused of market competition, monopolization involving ANY tech companies, or anti-competitive practices by major platforms or AI companies.
2. IP Law: refers to cases where the defendant is accused of patents, copyrights, trademarks for AI models or tech, or training data disputes, AI-generated content ownership.
3. Privacy and Data Protection: refers to cases where the defendant is accused of data breaches, unauthorized data collection by automated systems, or privacy violations involving algorithms or data processing.
4. Tort: refers to cases where the defendant is accused of physical harm, emotional distress, negligence involving ANY automated systems, or defamation, personal injury from tech systems or algorithms.
5. Justice and Equity: refers to cases where the defendant is accused of discrimination or bias **caused by AI, automated systems, or algorithms** (e.g., hiring, lending, search). Do not use this category for discrimination cases without automation.
6. Consumer Protection: refers to cases where the defendant is accused of deceptive practices, unfair business practices with tech/automated systems, or misleading marketing of tech products or AI capabilities.
7. AI in Legal Proceedings: refers to cases where AI systems are merely used in the court processes, legal case management, or litigation tools. The core contention is not about AI, but AI tools have been used in the litigation process.
8. Unrelated: refers to cases that have no meaningful connection to artificial intelligence (AI), machine learning (ML), or automated systems. If the case involves discrimination, privacy, or other issues **without automation/AI/algorithmic involvement**, classify as Unrelated.

Rule 1: Classify the cases from the categories above on the aspect of the case specifically related to AI (i.e., AI in Legal Proceedings, Antitrust, Consumer Protection, IP Law, Tort, Justice and Equity, Unrelated)
Rule 2: If multiple categories apply, use all relevant categories. If no category applies, use "Unrelated".
Rule 3: Respond with JSON: {"category": ["category_name1", "category_name2", ...]}.'''


def get_raw_response(prompt, model="gpt-4o-mini", **kwargs):
    with open("otherkey.txt") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                **kwargs
            )
            msg = response.choices[0].message
            content = msg.content or ""
            print("pluh: " + content + "")
            return content
        except Exception as e:
            if "Rate limit" in str(e) or "429" in str(e):
                sleep_time = (2 ** attempt) + 5
                print(f"Rate limit hit. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def classify_text(title, text, model="gpt-4o-mini"):
    print(f"Classifying: {title}")
    prompt = CLASSIFY_INSTRUCTIONS.replace("{title}", title).replace("{text}", text)

    raw = get_raw_response(prompt, model=model, temperature=0)
    try:
        return parse_output(raw)
    except Exception:
        convert_prompt = (
            "Convert the following content into ONLY valid JSON with this exact schema:\n"
            '{"category": ["category_name1", "category_name2", ...]}\n'
            "Do not add any extra keys, commentary, or code fences.\n\n"
            "Content:\n<<<\n" + raw + "\n>>>"
        )
        raw2 = get_raw_response(convert_prompt, model=model, temperature=0)
        return parse_output(raw2)


def extract_json_string(s: str) -> str:
    """
    Extract the first JSON object or array from a string.
    Handles:
      - ```json ... ```
      - ``` ... ```
      - Prose with an embedded { ... } object or [ ... ] array
    """
    if not s:
        raise ValueError("Empty response; no JSON to parse.")

    # ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # ``` ... ```
    m = re.search(r"```\s*([\s\S]*?)\s*```", s)
    if m:
        candidate = m.group(1).strip()
        if (candidate.startswith("{") and candidate.endswith("}")) or \
           (candidate.startswith("[") and candidate.endswith("]")):
            return candidate

    def find_balanced(text, open_ch, close_ch):
        start = text.find(open_ch)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        i = start
        while i < len(text):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            i += 1
        return None

    obj = find_balanced(s, "{", "}")
    if obj is not None:
        return obj
    arr = find_balanced(s, "[", "]")
    if arr is not None:
        return arr

    raise ValueError("Could not find a balanced JSON object or array in the response.")


def parse_output(output: str) -> dict:
    json_str = extract_json_string(output)
    data = json.loads(json_str)

    if isinstance(data, list):
        return {"category": data}

    if isinstance(data, dict):
        for k, v in list(data.items()):
            norm = k.strip().lower().rstrip(":")
            if norm in ("category", "categories"):
                return {"category": v if isinstance(v, list) else [v]}

        for v in data.values():
            if isinstance(v, list):
                return {"category": v}

    raise KeyError("category")


def main(DATA_PATH):
    # Input format: {"casename": "case summary", ...}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, dict):
        raise ValueError("Input JSON must be an object mapping casename -> case summary")

    case_to_categories = {}

    for title, summary_text in cases.items():
        try:
            result = classify_text(title, summary_text)
            case_to_categories[title] = result
        except Exception as e:
            print(f"[WARN] Failed to classify '{title}': {e}")
            case_to_categories[title] = {"error": str(e)}

    with open("categorized_cases.json", "w", encoding="utf-8") as f:
        json.dump(case_to_categories, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/data.json")
        sys.exit(1)
    data_path = sys.argv[1]
    main(data_path)
