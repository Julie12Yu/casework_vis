# similar approach to 4_actors_breakdown.py, use some kind of rule-based classifier
import json
from collections import Counter
import re
from pathlib import Path
import spacy
from rake_nltk import Rake


nlp = spacy.load("en_core_web_sm")
rake = Rake()

NAME = "privacy"

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
        t = search_text.lower()

        # FACIAL RECOGNITION
        if any(k in t for k in [
            "facial geometry", "face recognition", "facial recognition", "facial features", "face data",
            "facial-recognition", "bipa", "biometric information", "biometric identifiers", "biometric",
            "facial image analysis"
        ]):
            return "face recognition"
            
        # GENERATIVE AI (LLM, copyright, creative use)
        if any(k in t for k in [
            "large language model", "llm", "generative ai", "generative artificial intelligence",
            "ai-based photo art", "chatgpt", "claude", "ai image-generation models", "image-generation model",
            "ai image generator", "large language", "ai-generated artwork", "book", "movie", "artwork", "music",
            "copyright law", "legal author", "dabus", "agi technology", "in re mosaic llm litigation",
            "ai-based legal research"
        ]):
            return "generative ai"

        # AI BOTS / CHATBOTS / ASSISTANTS
        if any(k in t for k in [
            "chatbot", "virtual assistant", "ai bot", "ai chatbot", "conversational ai",
            "ai-powered conversation intelligence", "ai-powered virtual pet", "ai-powered chatbot",
            "chat data", "ai assistant", "automated ai-like bots", "alexa", "siri", "smart speaker",
            "voice assistant", "google assistant", "ai phone assistants"
        ]):
            return "ai bots"
        
        # TRANSCRIPTION / SPEECH-TO-TEXT AI
        if any(k in t for k in [
            "speech-to-text", "transcription", "audio transcription", "speech recognition",
            "dictation", "meeting transcription", "video transcription", "natural language interface",
            "automated texting", "nlp", "automatic speech recognition", "asr", "voice-to-text",
        ]):
            return "transcription ai"


        # MEDICAL / HEALTHCARE AI
        if any(k in t for k in [
            "radiology", "medical imaging", "diagnostic ai", "imaging solutions",
            "clinical workflow", "x-ray", "mri", "ct scan", "ultrasound", "ai medical assistant",
            "patient data", "healthcare software", "physician tool", "ai radiology",
            "dental x-ray", "xrays", "medical image analysis"
        ]):
            return "medical ai"

        # COMPUTER VISION & IMAGE SYSTEMS
        if any(k in t for k in [
            "computer vision", "image analysis", "dashcam", "3d scene", "3d interior", "ar/vr",
            "vision analysis", "image detection", "image comparison", "object detection",
        ]):
            return "computer vision ai"

        # ALGORITHIM / ANALYTICAL AI / ML Systems
        if any(k in t for k in [
            "algorithm", "algorithms", "algorithmic", "algorithmic decision-making", "predictive", "predict", 
            "ai system to target", "ai system to predict", "machine-learning models", "ml workloads", 
            "ai applicant-screening system", "hiring software", "age discrimination", "disparate impact",
            "property risk", "ml-based crm", "data analytics", "sound processing", "recruitment software", 
            "automated repair estimates", "vehicle damage assessment", "ai-powered ad tools", 
            "ai-enabled host websites", "ai-based ad fraud", "ad fraud detection", "invalid traffic detection", 
            "predictive fraud modeling", "benefit-decision models"
        ]):
            return "algorithims"

        # AUTONOMOUS / VEHICLE AI
        if any(k in t for k in [
            "autonomous vehicle", "autonomous driving", "dashcam design", "dashcam technology", 
            "driver safety camera", "fleet safety cameras", "vehicle automation software", "telematics solutions",
            "ai-based vehicle automation", "ai-driven dashboard camera"
        ]):
            return "autonomous/vehicle ai"      

        # CYBERSECURITY AI
        if "cybersecurity" in t:
            return "cybersecurity"

        # TRADE SECRETS / PROPRIETARY AI
        if any(k in t for k in [
            "trade secrets", "misappropriation", "proprietary ai", "ai technology forms core alleged trade secrets",
            "ai code", "confidential ai-related information", "ai/ml source code"
        ]):
            return "ai trade secrets"

        # ENTERPRISE AI — business & operational AI systems
        if any(k in t for k in [
            "ediscovery", "e-discovery", "crm", "workflow", "enterprise",
            "document review", "customer management", "business analytics",
            "business intelligence", "marketing ai", "sales ai"
        ]):
            return "enterprise ai"

        # QUINTUS
        if any(k in t for k in ["quintus"]):
            return "quintus"

        # AI PATENT / IP DISPUTES
        if "patent" in t:
            return "ai patent / ip"

        # otherwise none
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

        ai_presence = case.get("ai_presence", "")
        ai_relevance = case.get("ai_relevance", "")
        plaintiff_desc = parties.get("plaintiff_description", "")
        defendant_desc = parties.get("defendant_description", "")

        # Combine all relevant text into one classification input
        if ai_relevance == "NOT RELATED":
            ai_relevance = ""
        combined_text = " ".join([
            ai_presence,
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