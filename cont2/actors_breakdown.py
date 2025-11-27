import json
from collections import Counter
import re
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "privacy/priv_cases_breakdown.json"
OUTPUT_PATH = "privacy/actor_analysis.json"

# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text


def extract_party_type(description):
    """
    A far more detailed classifier to reduce 'other' occurrences.
    Uses keyword rules to infer entity type from description.
    """

    text = normalize_text(description)

    # INDIVIDUAL / PEOPLE
    if any(k in text for k in [
        "individual", "person", "employee", "worker", "citizen", "resident",
        "plaintiff is a", "former employee", "job applicant", "customer"
    ]):
        return "individual"

    # CORPORATIONS / COMPANIES
    if any(k in text for k in [
        "corporation", "company", "inc", "llc", "l.l.c.", "ltd", "limited",
        "corp", "business", "employer", "restaurant", "store",
        "consulting firm", "insurance company", "retail chain", "tech company"
    ]):
        return "corporation"

    # NONPROFITS / FOUNDATIONS / ASSOCIATIONS
    if any(k in text for k in [
        "nonprofit", "foundation", "charity", "organization", "association",
        "advocacy group", "coalition"
    ]):
        return "nonprofit organization"

    # PARTNERSHIPS
    if any(k in text for k in [
        "partnership", "general partnership", "limited partnership", "llp"
    ]):
        return "partnership"

    # HEALTHCARE / HOSPITALS
    if any(k in text for k in [
        "hospital", "clinic", "medical center", "health system", "healthcare",
        "physician group"
    ]):
        return "healthcare provider"

    # EDUCATIONAL INSTITUTIONS
    if any(k in text for k in [
        "university", "college", "school", "school district",
        "academy", "board of education"
    ]):
        return "educational institution"

    # BANKS / FINANCIAL
    if any(k in text for k in [
        "bank", "credit union", "financial institution", "investment firm",
        "brokerage"
    ]):
        return "financial institution"

    # MEDIA / PUBLISHERS
    if any(k in text for k in [
        "newspaper", "media company", "publisher", "broadcasting",
        "television", "news organization"
    ]):
        return "media organization"

    # UNIONS
    if any(k in text for k in [
        "union", "labor union", "local chapter", "collective bargaining unit"
    ]):
        return "labor union"

    # INSURANCE COMPANIES (special category)
    if "insurance" in text:
        return "insurance company"

    # LAW ENFORCEMENT
    if any(k in text for k in [
        "police", "sheriff", "law enforcement", "detective", "state trooper",
        "patrol", "police department"
    ]):
        return "law enforcement agency"

    # GOVERNMENT ENTITIES (broad)
    if any(k in text for k in [
        "government", "govt", "state of", "county of", "city of",
        "municipality", "public agency", "department", "bureau",
        "division of", "state agency", "public authority"
    ]):
        return "government entity"

    # MUNICIPALITIES (more specific)
    if any(k in text for k in [
        "city", "township", "borough", "village", "town", "municipality"
    ]):
        return "municipality"

    # FEDERAL AGENCIES
    if any(k in text for k in [
        "federal", "u.s. department", "united states department",
        "us department", "homeland security", "fbi", "irs", "department of"
    ]):
        return "federal agency"

    # FOREIGN GOVERNMENTS
    if any(k in text for k in [
        "embassy", "consulate", "foreign ministry", "government of"
    ]):
        return "foreign government"

    # DEFAULT
    return "other"



# ============================================================
# MAIN ANALYSIS (ACTORS ONLY)
# ============================================================

def analyze_actors(cases):
    plaintiff_types = Counter()
    defendant_types = Counter()
    combinations = Counter()

    simplified_output = []

    for case in cases:
        parties = case.get("parties", {})

        p_name = parties.get("plaintiff_name", "")
        p_desc = parties.get("plaintiff_description", "")
        d_name = parties.get("defendant_name", "")
        d_desc = parties.get("defendant_description", "")

        p_type = extract_party_type(p_desc)
        d_type = extract_party_type(d_desc)

        plaintiff_types[p_type] += 1
        defendant_types[d_type] += 1
        combinations[f"{p_type} v. {d_type}"] += 1

        simplified_output.append({
            "case_id": case.get("case_id"),
            "case_name": case.get("case_name"),
            "ai_relevance": case.get("ai_relevance"),

            "plaintiff_name": p_name,
            "plaintiff_description": p_desc,
            "plaintiff_type": p_type,

            "defendant_name": d_name,
            "defendant_description": d_desc,
            "defendant_type": d_type
        })

    final_data = {
        "summary": {
            "total_cases": len(cases),
            "plaintiff_types": dict(plaintiff_types),
            "defendant_types": dict(defendant_types),
            "combinations": dict(combinations)
        },
        "cases": simplified_output
    }

    return final_data

def main():
    print("\n📘 ACTOR-ONLY CASE ANALYSIS")
    print("=" * 70)

    # Load cases
    with open(INPUT_PATH, "r") as f:
        cases = json.load(f)

    print(f"✓ Loaded {len(cases)} cases")

    results = analyze_actors(cases)

    # Save output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Actor-only analysis saved to {OUTPUT_PATH}")
    print(f"✓ Completed successfully.")


if __name__ == "__main__":
    main()