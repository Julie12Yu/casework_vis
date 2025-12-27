import json
from collections import Counter
import re
from pathlib import Path
import spacy
from rake_nltk import Rake


nlp = spacy.load("en_core_web_sm")
rake = Rake()

NAME = "ipLaw"

# inconsistent usage for now
#INPUT_PATH = f"labeled_data/{NAME}/{NAME}/cases_breakdown.json"
#OUTPUT_PATH = f"labeled_data/{NAME}/{NAME}/actor_analysis.json"
 
INPUT_PATH = f"raw_data/0_llm_run.txt"
OUTPUT_PATH = f"raw_data/0_llm_analysis.json"

def extract_ai_tech_type(description):
    """
    1. RAKE -> Get phrases that matter
    2. Use rule-based classifier
    """

    # Normalize fallback text
    text = description

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

    # SECOND PASS: Check full normalized text
    result = check_rules(text)
    if result:
        return result

    # OTHER
    return text


def analyze_actors(cases):
    plaintiff_types = Counter()
    defendant_types = Counter()
    plaintiff_categories = Counter()
    defendant_categories = Counter()
    tech_types = Counter()

    simplified_output = []

    for case in cases:
        case_name = case.get("title", "")

    #        categories = case.get("categories", [])  # ngl idk what to do with this
    # i think will be changed soon

        plaintiff = case.get("plaintiff", {})
        defendant = case.get("defendant", {})
        raw_tech_used = case.get("core_AI_system", {})

        index = 0
        text = None
        while text is None and (index < len(raw_tech_used)):
            text = extract_ai_tech_type(raw_tech_used[index])
            index += 1
        
        tech_types[text] += 1 

        plaintiff_raw_type = plaintiff.get("entity_type", "")

        defendant_raw_type = defendant.get("entity_type", "")

        plaintiff_types[plaintiff_raw_type] += 1
        defendant_types[defendant_raw_type] += 1

    final_data = {
        "total_cases": len(cases),
        "plaintiff_types": dict(plaintiff_types),
        "defendant_types": dict(defendant_types),
        "tech_types": dict(tech_types)
    }

    return final_data


def main():
    print("\n📘 ACTOR-ONLY CASE ANALYSIS")
    print("=" * 70)

    # txt -> json -> use it
    cases = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            line = line.strip() 
            cases.append(json.loads(line))
    

    print(f"✓ Loaded {len(cases)} cases")

    results = analyze_actors(cases)

    # Save output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Actor-only analysis saved to {OUTPUT_PATH}")
    print(f"✓ Completed successfully.")


if __name__ == "__main__":
    main()