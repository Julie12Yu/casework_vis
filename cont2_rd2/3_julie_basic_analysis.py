import json
from collections import Counter
import re
from pathlib import Path
import spacy
from rake_nltk import Rake


nlp = spacy.load("en_core_web_sm")
rake = Rake()

NAME = "privacy"

INPUT_PATH = f"raw_data/base_raw/relevant_cases_breakdown.json"
OUTPUT_PATH = f"raw_data/relevant_all_analysis.json"
OUTPUT_PATH_2 = f"raw_data/relevant_cases_analyzed.json"

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text


def extract_party_type(description):
    """    
    1. RAKE -> Get phrases that matter
    2. Use rule-based classifier
    3. NER -> On leftovers, if person, return individual
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
        # CLASS ACTION
        if any(k in search_text for k in [
            "class action", "class-action", "putative class", "nationwide class", "class of "
        ]):
            return "class-action"
        
        # STEM CREATIVES
        if any(k in search_text for k in [
            "developer", "engineer", "scientist", "programmer", "technologist",
            "software creator", "software engineer", "data scientist", "ai researcher",
            "inventor", "researcher"
        ]):
            return "individual"

        # ART CREATIVES
        if any(k in search_text for k in [
            "author", "director", "editor", "illustrator", "composer", "songwriter", "artist", 
            "poet", "musician", "writer", "filmmaker", "designer", "photographer", "content creator"
        ]):
            return "individual"

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
        
        # FINANCE COMPANIES?
        if any(k in search_text for k in [
            "finance company", "financial institution", "bank", "lender"
        ]):
            return "finance company"
        
        # GOVERNMENT ENTITIES (broad)
        if any(k in search_text for k in [
            "government", "govt", "state of", "county of", "city of",
            "municipality", "public agency", "department", "bureau",
            "division of", "state agency", "public authority", "federal agency"
        ]):
            return "government entity"

        # INDIVIDUAL with an PUBLISHER
        if any(k in search_text for k in [
            "publisher", "publishing company", "publishing entity", "rightsholders"
        ]):
            return "individual with publisher"

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

    ###
    # 3. NER: FIND INDIVIDUALS
    doc = nlp(description)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return "individual"

    # OTHER
    return text


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
            
        # GENERATIVE AI (LLM, copyright, creative use, text-to-image, deepfakes)
        if any(k in t for k in [
            "large language model", "llm", "generative ai", "generative artificial intelligence", "image-generation",
            "ai-based photo art", "chatgpt", "claude", "ai image-generation models", "image-generation model",
            "ai image generator", "large language", "ai-generated artwork", "copyright law", "legal author",
            "dabus", "agi technology", "in re mosaic llm litigation", "ai-based legal research",
            "stable diffusion", "midjourney", "deviantart", "text-to-image", "image generation",
            "diffusion model", "image synthesis", "ai-generated content", "deepfake", "deep fake",
            "hallucinated", "hallucination", "synthetic media", "playground ai"
        ]):
            return "generative ai"

        # AUTOMATED CALLING SYSTEMS (robocalls, autodialers, avatars, spoofing)
        if any(k in t for k in [
            "automated telemarketing", "prerecorded call", "prerecorded message",
            "robocall", "robodial", "autodialer", "automatic telephone dialing system",
            "atds", "soundboard technology", "avatar voice", "ai-generated voice",
            "spoofed caller id", "call campaign", "telemarketing system", "dialing platform",
            "pre-recorded sales call", "automated dialing system", 
        ]):
            return "automated calling systems"

        # AI BOTS / CHATBOTS / ASSISTANTS
        if any(k in t for k in [
            "chatbot", "virtual assistant", "ai bot", "ai chatbot", "conversational ai",
            "ai-powered conversation intelligence", "ai-powered virtual pet", "ai-powered chatbot",
            "chat data", "ai assistant", "automated ai-like bots", "alexa", "siri", "smart speaker",
            "voice assistant", "google assistant", "ai phone assistants",
            "phone-order assistant", "npc", "non-player character", "website bot", "embedded chat"
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
            "dental x-ray", "xrays", "medical image analysis", "precision medicine",
            "personalized medicine", "drug discovery", "clinical trials", "seizure detection"
        ]):
            return "medical ai"

        # COMPUTER VISION & IMAGE SYSTEMS
        if any(k in t for k in [
            "computer vision", "computer-vision", "image analysis", "dashcam", "3d scene", "3d interior", "ar/vr",
            "vision analysis", "image detection", "image comparison", "object detection",
            "proctoring", "exam monitoring", "tracking", "rockslide detection", "drone imaging",
            "virtual reality"
        ]):
            return "computer vision ai"
        

        # PLATFORM MODERATION
        if any(k in t for k in [
            "platform moderation", "integrity score", "content moderation", 
            "automated moderation", "automated enforcement", "ai moderation",
            "harmful content detection", "policy violation detection",
            "automated takedown", "flagged content", "content filtering",
            "account suspension", "account enforcement", "automated ban",
            "bot detection system", "fake account detection", "spam detection engine",
            "hate speech detection", "misinformation detection",
            "harassment detection", "abusive content classifier",
            "child safety classifier", "csam detection", "ai moderation"
        ]):
            return "recommendation"
        
        # AI hiring & employment screening
        if any(k in t for k in [
            "employability score", "hiring decision assisting ai", "hiring"
            "interview analytics", "resume screening", "talent screening",
            "hiring ai"
        ]):
            return "decision"

        # --- DECISION / RISK / ELIGIBILITY SYSTEMS ---
        decision_keywords = [
            # financial / risk / insurance / claims
            "risk scoring", "risk assessment", "risk model",
            "underwriting", "credit scoring", "loan decision",
            "loan approval", "claims adjudication", "claims review",
            "claims denial", "medical necessity", "coverage determination",
            "prior authorization", "insurance pricing", "actuarial model",
            "home-pricing algorithm", "forecasting algorithm",
            # fraud / pricing / allocation / blocking
            "fraud detection", "fraud scoring", "predictive fraud",
            "pricing optimization", "dynamic pricing",
            "ad pricing", "ad distribution control",
            "ad review", "ad rejection", "content blocking",
            "automated moderation", "enforcement decision",
            # resources / eligibility / government
            "eligibility determination", "benefits eligibility",
            "automated eligibility", "automated adjudication",
            "visa revocation", "catch and revoke",
            # trading / investment
            "algorithmic trading", "trading strategy",
            "forex trading", "automated trading",
            "evaluate investment opportunities",
            # healthcare / scoring / prediction tied to access
            "coverage decision", "coverage tool", "utilization management",
            "predictive model scoring medical necessity",
            # safety / risk classification
            "driver monitoring", "distracted driving prediction",
            # damage / repair cost decisions
            "repair cost estimate", "vehicle damage estimate",
            # employment / screening decisions
            "resume screening", "hiring screening", "hiring decision",
            # content approvals that determine access, not ranking
            "review and reject advertisements",
            # catch-all
            "decision engine", "decision model", "allocation system",
            "resource allocation"
        ]

        if any(k in t for k in decision_keywords):
            return "decision"

        # --- RECOMMENDATION / RANKING / TARGETING / PERSONALIZATION ---
        recommendation_keywords = [
            "recommendation system", "recommendation engine",
            "ranking system", "algorithmic ranking",
            "content ranking", "feed ranking",
            "news feed", "video feed", "for you page",
            "personalization", "personalized results",
            "suggested content", "suggested videos",
            "content recommendation", "search ranking",
            "auto-recommend", "auto-suggest", "curation system",
            "product recommendations", "similar items", "users also liked",
            "playlist generation",
            # matching platforms
            "match home buyers", "match buyers", "matching buyers",
            "match businesses with", "matching professionals",
            # targeted ads / distribution choices (not approvals)
            "ad targeting", "optimize ad delivery",
            "ad distribution optimization",
            # ecommerce optimization / choosing products
            "select profitable products", "product selection algorithm",
            "inventory recommendation",
            # engagement optimization
            "optimize engagement", "optimize gameplay", "reinforce gameplay",
            "optimize user experience",
            # personalization in consumer services
            "customizes skincare", "personalized skincare",
            "personalized education", "personalized tutoring",
            # diversity / representation controls
            "ensure diverse content representation",
            # content discovery tools
            "discover content", "content discovery",
            # real estate discovery platforms
            "connect home buyers with properties",
            # data labeling / annotation guidance
            "content-annotation", "data-labeling automation"
        ]

        if any(k in t for k in recommendation_keywords):
            return "recommendation"

        # AUTONOMOUS / VEHICLE AI
        if any(k in t for k in [
            "autonomous vehicle", "autonomous driving", "dashcam design", "dashcam technology",
            "driver safety camera", "fleet safety cameras", "vehicle automation software", "telematics solutions",
            "ai-based vehicle automation", "ai-driven dashboard camera", "autonomous boat", "autonomous drone"
        ]):
            return "autonomous/vehicle ai"

        # CYBERSECURITY AI
        if "cybersecurity" in t:
            return "cybersecurity"

        # TRADE SECRETS / PROPRIETARY AI
        if any(k in t for k in [
            "trade secrets", "misappropriation", "proprietary ai",
            "ai technology forms core alleged trade secrets", "ai code",
            "confidential ai-related information", "ai/ml source code"
        ]):
            return "ai trade secrets"

        # ENTERPRISE AI
        if any(k in t for k in [
            "ediscovery", "e-discovery", "crm", "workflow", "enterprise",
            "document review", "customer management", "business analytics",
            "business intelligence", "marketing ai", "sales ai",
            "devops", "software development platform", "project management",
            "workforce analytics", "finance and accounting platform", "saas"
        ]):
            return "enterprise ai"

        # AI legal research and drafting
        if any(k in t for k in [
            "ai-powered legal research", "automated legal research",
            "ai-generated citations", "ai drafting tool", "ross intelligence"
        ]):
            return "ai legal research"

        # AI financial trading & credit decisioning
        if any(k in t for k in [
            "automated trading", "ai trading platform", "algorithmic trading",
            "credit decisioning", "loan underwriting platform"
        ]):
            return "ai finance systems"

        # AI proctoring & academic integrity
        if any(k in t for k in [
            "remote proctoring", "exam proctoring", "cheating detection",
            "room scan", "academic integrity system", "hirevue", "video interview"
        ]):
            return "ai proctoring"

        # AI e-commerce automation
        if any(k in t for k in [
            "product selection ai", "dropshipping automation", "e-commerce automation",
            "store automation", "ai product finder"
        ]):
            return "ai e-commerce automation"

        # AI energy optimization
        if any(k in t for k in [
            "energy storage optimization", "clean energy optimization",
            "grid optimization", "battery optimization"
        ]):
            return "ai energy optimization"

        # AI education / tutoring & content
        if any(k in t for k in [
            "ai tutoring", "ai classroom", "educational ai",
            "learning platform ai", "virtual teacher"
        ]):
            return "ai education"

        # AI law enforcement / surveillance analytics
        if any(k in t for k in [
            "visa revocation program", "law enforcement ai",
            "surveillance analytics", "threat detection ai"
        ]):
            return "ai law enforcement"

        # QUINTUS
        if "quintus" in t:
            return "quintus"

        # AI PATENT / IP DISPUTES
        if "patent" in t or "trademark" in t or "ip " in t:
            return "ai patent / ip"

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
        case_name = case.get("case_id", "")

    #        categories = case.get("categories", [])  # ngl idk what to do with this
    # i think will be changed soon

        plaintiff = case.get("plaintiff", "")
        defendant = case.get("defendant", "")
        raw_tech_used = case.get("core_ai_system", "")
        plaintiff_arg_labels = case.get("plaintiff_labels", {})
        defendant_arg_labels = case.get("defendant_labels", {})

        text = extract_ai_tech_type(raw_tech_used)
        
        tech_types[text] += 1 

        p_entity_type = (
            plaintiff.get("entity_type", "") if isinstance(plaintiff, dict) else plaintiff
        )
        d_entity_type = (
            defendant.get("entity_type", "") if isinstance(defendant, dict) else defendant
        )

        plaintiff_raw_type = extract_party_type(plaintiff.get("entity_type", ""))

        defendant_raw_type = extract_party_type(defendant.get("entity_type", ""))

        plaintiff_types[plaintiff_raw_type] += 1
        defendant_types[defendant_raw_type] += 1

        # ---- Store per-case result ----
        simplified_output.append({
            "title": case_name,
            "plaintiff_raw": p_entity_type,
            "plaintiff_category": plaintiff_raw_type,
            "defendant_raw": d_entity_type,
            "defendant_category": defendant_raw_type,
            "tech_raw": raw_tech_used,
            "tech_category": text,
            "plaintiff_arg_labels": plaintiff_arg_labels,
            "defendant_arg_labels": defendant_arg_labels,
        })

    final_data = {
        "total_cases": len(cases),
        "plaintiff_types": dict(plaintiff_types),
        "defendant_types": dict(defendant_types),
        "tech_types": dict(tech_types)
    }

    cases_new = {
        "cases": simplified_output
    }

    return final_data, cases_new


def main():
    print("\n📘 ACTOR-ONLY CASE ANALYSIS")
    print("=" * 70)

    # txt -> json -> use it
    with open(INPUT_PATH, "r") as f:
        cases = json.load(f)
    

    print(f"✓ Loaded {len(cases)} cases")

    final_data, cases_new = analyze_actors(cases)

    # Save output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_data, f, indent=2)

    with open(OUTPUT_PATH_2, "w") as f:
        json.dump(cases_new, f, indent=2)

    print(f"✓ Completed successfully.")


if __name__ == "__main__":
    main()