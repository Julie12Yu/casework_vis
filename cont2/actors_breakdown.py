import json
from collections import Counter
import re
from pathlib import Path
import spacy
from rake_nltk import Rake


nlp = spacy.load("en_core_web_sm")
rake = Rake()

INPUT_PATH = "ipLaw/cases_breakdown.json"
OUTPUT_PATH = "ipLaw/actor_analysis.json"


def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text


def extract_party_type(description):
    """
    1. NER -> If person, return individual
    2. RAKE -> Get phrases that matter
    3. Use rule-based classifier
    """

    # Normalize fallback text
    text = normalize_text(description)

    ###
    # 1. NER: FIND INDIVIDUALS
    doc = nlp(description)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return "individual"

    ###
    # 2. RAKE Keyphrase Extraction
    rake.extract_keywords_from_text(description)
    phrases = [p.lower() for p in rake.get_ranked_phrases()]
    phrase_text = " ".join(phrases)

    ###
    # 3. RULE-BASED CLASSIFICATION on keyphrases
    def check_rules(search_text):
        # CLASS ACTION
        if any(k in search_text for k in [
            "class action", "class-action", "putative class", "nationwide class", "class of "
        ]):
            return "class-action"

        # PLATFORMS (e.g., social media, marketplaces, they function different legally from companies)
        if any(k in search_text for k in [
            "social media platform", "online marketplace", "e-commerce platform",
            "social networking site", "content sharing platform", "video sharing platform",
            "platform operator", "digital platform", "online platform", "platform", "platforms"
        ]):
            return "platform(s)"

        # CORPORATIONS / COMPANIES
        if any(k in search_text for k in [
            "corporation", "company", "inc", "llc", "l.l.c.", "ltd", "limited",
            "corp", "business", "employer", "restaurant", "store",
            "consulting firm", "insurance company", "retail chain", "tech company",
            "companies"
        ]):
            return "corporation(s)"

        # NONPROFITS / CHARITIES
        if any(k in search_text for k in [
            "nonprofit", "charity", "advocacy group"
        ]):
            return "nonprofit organization"

        # INVESTORS
        if any(k in search_text for k in [
            "investor", "investors", "shareholder", "venture capital", "stock purchasers"
        ]):
            return "investor(s)"

        # HEALTHCARE / HOSPITALS
        if any(k in search_text for k in [
            "hospital", "clinic", "medical center", "health system", "healthcare",
            "physician group"
        ]):
            return "healthcare provider"

        # INSURANCE COMPANIES
        if any(k in search_text for k in [
            "insurance company", "insurer", "insurance provider", "insurance carrier"
        ]):
            return "insurance company"
        
        # GOVERNMENT ENTITIES (broad)
        if any(k in search_text for k in [
            "government", "govt", "state of", "county of", "city of",
            "municipality", "public agency", "department", "bureau",
            "division of", "state agency", "public authority", "federal agency"
        ]):
            return "government entity"

        # INDIVIDUAL / PEOPLE
        if any(k in search_text for k in [
            "individual", "person", "employee", "worker", "citizen", "resident",
            "plaintiff is a", "former employee", "job applicant", "customer", 
            "sex trafficking victim",
        ]):
            return "individual"
        return None

    # FIRST PASS: Check phrase_text
    result = check_rules(phrase_text)
    if result:
        return result

    # SECOND PASS: Check full normalized text
    result = check_rules(text)
    if result:
        return result

    # OTHER
    return text


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