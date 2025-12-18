#!/usr/bin/env python3
import json
from collections import Counter

INPUT_JSON = "../misc/new_court_cases_processed.json"


def load_documents(input_json):
    print(f"Loading processed data from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    docs = obj["documents"]
    meta = obj.get("meta", {})
    print(f"  Loaded {len(docs)} documents")
    return docs, meta


def count_categories(docs):
    # Count by numeric ID and by human-readable name
    counts_by_num = Counter(d.get("legal_category", None) for d in docs)
    counts_by_name = Counter(d.get("legal_category_name", "Unknown") for d in docs)

    return counts_by_num, counts_by_name


def main():
    docs, meta = load_documents(INPUT_JSON)
    counts_by_num, counts_by_name = count_categories(docs)

    # Optional: pull canonical category names from meta if present
    legal_categories_meta = meta.get("legal_categories", {})

    print("\nCases per legal category (by number):")
    for cat_num in sorted(k for k in counts_by_num.keys() if k is not None):
        name_from_meta = legal_categories_meta.get(str(cat_num)) or legal_categories_meta.get(cat_num)
        display_name = name_from_meta if name_from_meta else "Unknown"
        print(f"  {cat_num} ({display_name}): {counts_by_num[cat_num]}")

    print("\nCases per legal category (by name):")
    for name, count in sorted(counts_by_name.items(), key=lambda x: x[0]):
        print(f"  {name}: {count}")

    total = sum(counts_by_name.values())
    print(f"\nTotal documents: {total}")


if __name__ == "__main__":
    main()
