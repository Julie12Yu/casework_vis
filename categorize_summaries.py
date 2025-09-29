import json
import os
import time
from typing import Dict, Any, Optional, List

INPUT_FILE_PATH = "categories_from_summaries.json"
OUTPUT_FILE_PATH = "categories_sept_29.json"

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Config via env
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
MAX_SUMMARIES_PER_CLUSTER = int(os.getenv("MAX_SUMMARIES_PER_CLUSTER", "0")) # if non-zero, creates limit

try:
    from openai import OpenAI
    _OPENAI_SDK_V1 = True
except Exception:
    import openai as _openai_legacy
    _OPENAI_SDK_V1 = False


# System prompt
SYSTEM_PROMPT = (
    "You are a precise legal-tech classifier. Assign each CLUSTER (a group of case summaries) "
    "to exactly one category from this taxonomy, or 'Unrelated' if none apply. "
    "Be conservative: do NOT assign an AI-related category unless AI is CENTRAL to the dispute or decision. "
    "Briefly justify based on the dominant/common themes across the cluster. Return strict JSON.\n\n"
    "Allowed categories (keys):\n"
    "- Antitrust\n"
    "- IP Law\n"
    "- Privacy and Data Protection\n"
    "- Tort\n"
    "- Justice and Equity\n"
    "- Consumer Protection\n"
    "- AI in Legal Proceedings\n"
    "- Unrelated\n\n"
    "Definitions / cues:\n"
    "Antitrust: refers to cases where the defendant is accused of market competition, monopolization involving ANY tech companies, or anti-competitive practices by major platforms or AI companies.\n"
    "IP Law: refers to cases where the defendant is accused of patents, copyrights, trademarks for AI models or tech, or training data disputes, AI-generated content ownership.\n"
    "Privacy and Data Protection: refers to cases where the defendant is accused of data breaches, unauthorized data collection by automated systems, or privacy violations involving algorithms or data processing.\n"
    "Tort: refers to cases where the defendant is accused of physical harm, emotional distress, negligence involving ANY automated systems, or defamation, personal injury from tech systems or algorithms.\n"
    "Justice and Equity: refers to cases where the defendant is accused of discrimination or bias **caused by AI, automated systems, or algorithms** (e.g., hiring, lending, search). Do not use this category for discrimination cases without automation.\n"
    "Consumer Protection: refers to cases where the defendant is accused of deceptive practices, unfair business practices with tech/automated systems, or misleading marketing of tech products or AI capabilities.\n"
    "AI in Legal Proceedings: refers to cases where AI systems are merely used in the court processes, legal case management, or litigation tools. The core contention is not about AI, but AI tools have been used in the litigation process.\n"
    "Unrelated: refers to cases that have no meaningful connection to artificial intelligence (AI), machine learning (ML), or automated systems. If the case involves discrimination, privacy, or other issues **without automation/AI/algorithmic involvement**, classify as Unrelated.\n\n"
    "Output (strict JSON only): "
    "{\"category\": <one of the keys>, \"confidence\": number between 0 and 1, \"rationale\": short string}"
)

# User instruction template (cluster-level)
USER_INSTRUCTION_TEMPLATE = (
    "You will classify a CLUSTER of U.S. case SUMMARIES using the taxonomy above.\n"
    "- Choose exactly ONE category, or 'Unrelated'.\n"
    "- Base your decision on the dominant/common themes across the summaries (ignore outliers).\n"
    "- Keep the rationale concise and cluster-wide.\n\n"
    "CLUSTER NAME: {cluster_name}\n"
    "SUMMARIES (each item is one case in the cluster):\n{summaries_block}"
)


def _openai_client():
    if _OPENAI_SDK_V1:
        return OpenAI(api_key=API_KEY)
    return None


def _format_summaries_block(summaries: List[str]) -> str:
    lines = []
    for i, s in enumerate(summaries, 1):
        s = (s or "").strip()
        # Guard against runaway length per line
        if len(s) > 4000:
            s = s[:4000] + "…"
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


def classify_cluster_with_gpt(cluster_name: str, summaries: List[str]) -> Dict[str, Any]:
    if MAX_SUMMARIES_PER_CLUSTER and len(summaries) > MAX_SUMMARIES_PER_CLUSTER:
        use_summaries = summaries[:MAX_SUMMARIES_PER_CLUSTER]
    else:
        use_summaries = summaries

    payload_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_INSTRUCTION_TEMPLATE.format(
                cluster_name=cluster_name,
                summaries_block=_format_summaries_block(use_summaries),
            ),
        },
    ]

    backoff = 1.0
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if _OPENAI_SDK_V1:
                client = _openai_client()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=payload_messages,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                    timeout=REQUEST_TIMEOUT,
                )
                content = resp.choices[0].message.content
            else:
                _openai_legacy.api_key = API_KEY
                resp = _openai_legacy.ChatCompletion.create(
                    model=MODEL,
                    messages=payload_messages,
                    temperature=TEMPERATURE,
                    request_timeout=REQUEST_TIMEOUT,
                )
                content = resp["choices"][0]["message"]["content"]

            parsed = json.loads(content)
            category = parsed.get("category", "Unrelated")
            confidence = float(parsed.get("confidence", 0.0))
            rationale = str(parsed.get("rationale", ""))
            return {
                "category": category,
                "confidence": confidence,
                "rationale": rationale,
                "total_summaries": len(summaries),
                "used_summaries": len(use_summaries),
            }
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 8)

    return {
        "category": "Unrelated",
        "confidence": 0.0,
        "rationale": f"Classification failed after retries: {type(last_error).__name__ if last_error else 'Unknown error'}",
        "total_summaries": len(summaries),
        "used_summaries": len(use_summaries),
    }


def classify_clusters(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        data = json.load(f)

    # Load existing results to support resume-ability
    existing: Dict[str, Any] = {}
    if os.path.exists(OUTPUT_FILE_PATH):
        try:
            with open(OUTPUT_FILE_PATH, "r") as ef:
                existing = json.load(ef)
        except Exception:
            existing = {}

    results: Dict[str, Any] = existing if isinstance(existing, dict) else {}

    # Iterate clusters (buckets)
    for cluster_name, summaries in data.items():
        if not isinstance(summaries, list):
            continue

        # Skip if already classified
        if cluster_name in results and isinstance(results[cluster_name], dict):
            print(f"[{cluster_name}] Already classified; skipping.", flush=True)
            continue

        print(f"[{cluster_name}] Classifying cluster with {len(summaries)} summaries...", flush=True)
        cls = classify_cluster_with_gpt(cluster_name, summaries)
        print(f" -> {cls['category']} ({cls['confidence']:.2f})", flush=True)

        results[cluster_name] = {
            "category": cls["category"],
            "confidence": cls["confidence"],
            "rationale": cls["rationale"],
            "total_summaries": cls["total_summaries"],
            "used_summaries": cls["used_summaries"],
        }

        _atomic_write_json(OUTPUT_FILE_PATH, results)

    return results


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


if __name__ == "__main__":
    results = classify_clusters(INPUT_FILE_PATH)
    _atomic_write_json(OUTPUT_FILE_PATH, results)
    print(f"Wrote {OUTPUT_FILE_PATH}")