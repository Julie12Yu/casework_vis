import json
from collections import Counter
import re
from pathlib import Path
import matplotlib.pyplot as plt

TECH = "recommendation"
INPUT_PATH = "relevant_cases_breakdown.json"
OUTPUT_PATH = "cases_per_year.png"

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

def extract_year(title):
    m = re.search(r"(19|20)\d{2}", title)
    return int(m.group(0)) if m else None

def load_cases(path):
    with open(path, "r") as f:
        return json.load(f)

def count_by_year(cases, tech):
    c = Counter()
    for x in cases:
        y = extract_year(x.get("case_id", ""))
        tech_raw = x.get("core_ai_system", "")
        tech_fixed = extract_ai_tech_type(tech_raw)
        if y and tech_fixed == tech:
            c[y] += 1
    return c

def plot_hist(counts):
    years = sorted(counts.keys())
    values = [counts[y] for y in years]
    for year in years:
      print(f"year: {year}, count: {counts[year]}")

def main():
    cases = load_cases(INPUT_PATH)
    counts = count_by_year(cases, TECH)
    plot_hist(counts)

if __name__ == "__main__":
    main()
