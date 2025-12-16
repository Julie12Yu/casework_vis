# similar approach to 4_actors_breakdown.py, use some kind of rule-based classifier
import json
from collections import Counter
import re
from pathlib import Path
import spacy
from rake_nltk import Rake


nlp = spacy.load("en_core_web_sm")
rake = Rake()

NAME = "ipLaw"

INPUT_PATH = f"{NAME}/cases_breakdown.json"
OUTPUT_PATH = f"{NAME}/ai_tech.json"


def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text


def extract_ai_tech_type(description):
    """
    1. RAKE -> Get phrases that matter
    2. Use rule-based classifier
    """

    # Normalize fallback text
    text = normalize_text(description)

    ###
    # 1. RAKE Keyphrase Extraction
    rake.extract_keywords_from_text(description)
    phrases = [p.lower() for p in rake.get_ranked_phrases()]
    phrase_text = " ".join(phrases)

    ###
    # 2. RULE-BASED CLASSIFICATION on keyphrases
    def check_rules(search_text):
        #  FACIAL RECOGNITION
        if any(k in search_text for k in [
            "facial geometry", "face recognition", "facial recognition", "facial features", "face data",
            "facial-recognition", "bipa", "biometric information", "biometric identifiers", "biometric"
        ]):
            return "face recognition"
        
        # GENERATIVE AI
        if any(k in search_text for k in [
            "large language model", "llm", "generative ai", "generative artificial intelligence",
            "ai-based photo art", "chatgpt", "claude", "ai image-generation models", "image-generation model",
            "ai image generator", "large language", "ai-generated artwork", "book", "movie", "artwork", "music",
        ]):
            return "generative ai"
        
        # AI BOTS? (eg chatbots, virtual assistants)
        if any(k in search_text for k in [
            "chatbot", "virtual assistant", "ai bot", "ai chatbot", "conversational ai", "ai-powered conversation intelligence",
            "ai-powered virtual pet", "ai-powered chatbot", "chat data", "ai assistant", "automated ai-like bots"
        ]):
            return "ai bots"

        # ALGORITHIM / ANALYTICIAL AI?
        if any(k in search_text for k in [
            "algorithm", "algorithms", "algorithmic", "algorithmic decision-making", "predictive", "predict", "ai system to target",
            "ai system to predict"
        ]):
            return "algorithims"
        
        # CYBERSECURITY BACKED WITH AI
        if any(k in search_text for k in [
            "cybersecurity", "cyber security", "threat detection", "malware detection", "intrusion detection",
        ]):
            return "cybersecurity"
        
        # Qunitus. Holy moly.
        if any(k in search_text for k in [
            "Quintus", "quintus"
        ]):
            return "quintus"
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


def analyze_description(cases):
    """
    Case-level technology extraction using:
    - ai_relevance
    - plaintiff_description
    - defendant_description

    Output:
    [
      {
        "case_id": ...,
        "case_name": ...,
        "associated_technology": ...
      }
    ]
    """

    output = []
    technology_counter = Counter()

    for case in cases:
        parties = case.get("parties", {})

        ai_relevance = case.get("ai_relevance", "")
        plaintiff_desc = parties.get("plaintiff_description", "")
        defendant_desc = parties.get("defendant_description", "")

        # Combine all relevant text into one classification input
        combined_text = " ".join([
            ai_relevance,
            plaintiff_desc,
            defendant_desc
        ]).strip()

        # Extract associated technology using existing classifier
        associated_technology = extract_ai_tech_type(combined_text)

        technology_counter[associated_technology] += 1

        output.append({
            "case_id": case.get("case_id"),
            "case_name": case.get("case_name"),
            "associated_technology": associated_technology
        })

    return {
        "summary": {
            "total_cases": len(cases),
            "associated_technologies": dict(technology_counter)
        },
        "cases": output
    }



def main():
    print("\n📘 AI TYPE CASE ANALYSIS")
    print("=" * 70)

    # Load cases
    with open(INPUT_PATH, "r") as f:
        cases = json.load(f)

    print(f"✓ Loaded {len(cases)} cases")

    results = analyze_description(cases)

    # Save output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ AI type analysis saved to {OUTPUT_PATH}")
    print(f"✓ Completed successfully.")


if __name__ == "__main__":
    main()