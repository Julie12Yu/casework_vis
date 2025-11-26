import json
from collections import Counter, defaultdict
import re
from pathlib import Path
from difflib import SequenceMatcher

# Configuration
INPUT_PATH = "priv_cases_breakdown.json"
MAPPINGS_PATH = "term_mappings.json"
SUGGESTIONS_OUTPUT = "term_standardization_suggestions.json"
RAW_OUTPUT_PATH = "priv_cases_raw_output.json"

# Default mappings for common variations
DEFAULT_MAPPINGS = {
    # Party type standardization
    "individuals": "individual",
    "indivudals": "individual",
    "individual plaintiff": "individual",
    "individual defendant": "individual",
    "corporations": "corporation",
    "corp": "corporation",
    "corporate": "corporation",
    "company": "corporation",
    "companies": "corporation",
    "business": "corporation",
    "government": "government entity",
    "governmental": "government entity",
    "state": "government entity",
    
    # Common claim variations
    "age discrimination under adea": "age discrimination - adea",
    "age discrimination (adea)": "age discrimination - adea",
    "discrimination based on age": "age discrimination - adea",
    "adea violation": "age discrimination - adea",
    
    "invasion of privacy for publicizing": "invasion of privacy",
    "privacy violation": "invasion of privacy",
    "violation of privacy": "invasion of privacy",
    "publicizing private facts": "invasion of privacy",
    
    "retaliation under title vii": "retaliation - title vii",
    "title vii retaliation": "retaliation - title vii",
    "retaliation (title vii)": "retaliation - title vii",
    
    "disability discrimination under ada": "disability discrimination - ada",
    "ada violation": "disability discrimination - ada",
    "failure to accommodate disability": "disability discrimination - ada",
    
    "violation of njlad": "discrimination - njlad",
    "njlad violation": "discrimination - njlad",
    "discrimination under njlad": "discrimination - njlad",
}


def load_or_create_mappings():
    """Load existing mappings or create default"""
    if Path(MAPPINGS_PATH).exists():
        with open(MAPPINGS_PATH, "r") as f:
            mappings = json.load(f)
        print(f"✓ Loaded custom mappings from {MAPPINGS_PATH}")
        return mappings
    else:
        print("✓ Using default mappings (no custom mappings file found)")
        return DEFAULT_MAPPINGS.copy()


def save_mappings(mappings):
    """Save mappings for future use"""
    with open(MAPPINGS_PATH, "w") as f:
        json.dump(mappings, f, indent=2)
    print(f"✓ Saved mappings to {MAPPINGS_PATH}")


def normalize_text(text):
    """Normalize text for comparison"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text


def apply_mappings(text, mappings):
    """Apply mapping dictionary to normalize text"""
    normalized = normalize_text(text)
    return mappings.get(normalized, normalized)


def similarity_ratio(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()


def find_similar_terms(terms, threshold=0.85):
    """
    Find groups of similar terms that might be duplicates
    Returns suggestions for user review
    """
    suggestions = []
    checked = set()
    
    term_list = list(terms)
    
    for i, term1 in enumerate(term_list):
        if term1 in checked:
            continue
            
        similar_group = [term1]
        
        for term2 in term_list[i+1:]:
            if term2 in checked:
                continue
                
            # Check similarity
            ratio = similarity_ratio(term1, term2)
            
            if ratio >= threshold:
                similar_group.append(term2)
        
        if len(similar_group) > 1:
            # Sort by frequency if we have counts
            if isinstance(terms, dict):
                similar_group.sort(key=lambda x: terms[x], reverse=True)
            
            suggestions.append({
                "canonical": similar_group[0],  # Most common becomes canonical
                "variations": similar_group[1:],
                "similarity": "high" if ratio > 0.9 else "medium"
            })
            
            checked.update(similar_group)
    
    return suggestions


def extract_claim_type(claim_text):
    """Extract the main claim type from claim text"""
    claim_text = normalize_text(claim_text)
    
    patterns = [
        r'^([^-]+?)(?:\s+for\s+)',
        r'^([^-]+?)(?:\s+under\s+)',
        r'^([^-]+?)(?:\s+in violation of\s+)',
        r'^violation of\s+(.+?)(?:\s+for\s+|\s+by\s+|$)',
        r'^([^(]+?)(?:\s*\()',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, claim_text)
        if match:
            return normalize_text(match.group(1))
    
    words = claim_text.split()[:5]
    return ' '.join(words)


def extract_legal_basis_type(legal_basis):
    """Categorize legal basis"""
    legal_basis = normalize_text(legal_basis)
    
    if 'title vii' in legal_basis or 'title 7' in legal_basis:
        return 'title vii'
    elif 'adea' in legal_basis or 'age discrimination in employment act' in legal_basis:
        return 'adea'
    elif 'ada' in legal_basis and 'americans with disabilities' in legal_basis:
        return 'ada'
    elif 'njlad' in legal_basis or 'new jersey law against discrimination' in legal_basis:
        return 'njlad'
    elif 'u.s.c' in legal_basis or 'usc' in legal_basis:
        match = re.search(r'\d+\s*u\.?s\.?c\.?\s*§?\s*\d+', legal_basis)
        if match:
            return normalize_text(match.group(0))
        return 'federal statute'
    elif 'constitution' in legal_basis:
        return 'constitutional'
    elif 'common law' in legal_basis or 'tort' in legal_basis:
        return 'common law'
    elif 'state law' in legal_basis or any(state in legal_basis for state in ['new jersey', 'california', 'new york', 'texas']):
        return 'state law'
    else:
        return legal_basis[:40]


def analyze_and_suggest_standardization(cases):
    """
    Analyze cases and suggest term standardizations
    """
    print("\n" + "=" * 70)
    print("ANALYZING TERMS AND GENERATING STANDARDIZATION SUGGESTIONS")
    print("=" * 70)
    
    # Collect all unique terms
    all_claim_types = Counter()
    all_defense_types = Counter()
    all_plaintiff_types = Counter()
    all_defendant_types = Counter()
    all_legal_bases = Counter()
    
    for case in cases:
        # Claims
        for claim in case.get("claims", []):
            claim_type = extract_claim_type(claim.get("claim", ""))
            if claim_type:
                all_claim_types[claim_type] += 1
            
            legal_basis = extract_legal_basis_type(claim.get("legal_basis", ""))
            if legal_basis:
                all_legal_bases[legal_basis] += 1
        
        # Defenses
        for defense in case.get("defenses", []):
            defense_type = extract_claim_type(defense.get("defense", ""))
            if defense_type:
                all_defense_types[defense_type] += 1
        
        # Parties
        parties = case.get("parties", {})
        plaintiff_type = normalize_text(parties.get("plaintiff_type", ""))
        defendant_type = normalize_text(parties.get("defendant_type", ""))
        
        if plaintiff_type:
            all_plaintiff_types[plaintiff_type] += 1
        if defendant_type:
            all_defendant_types[defendant_type] += 1
    
    # Generate suggestions
    suggestions = {
        "claim_types": {
            "unique_count": len(all_claim_types),
            "total_instances": sum(all_claim_types.values()),
            "similar_groups": find_similar_terms(all_claim_types),
            "all_terms": dict(all_claim_types.most_common(50))
        },
        "defense_types": {
            "unique_count": len(all_defense_types),
            "total_instances": sum(all_defense_types.values()),
            "similar_groups": find_similar_terms(all_defense_types),
            "all_terms": dict(all_defense_types.most_common(50))
        },
        "plaintiff_types": {
            "unique_count": len(all_plaintiff_types),
            "total_instances": sum(all_plaintiff_types.values()),
            "similar_groups": find_similar_terms(all_plaintiff_types),
            "all_terms": dict(all_plaintiff_types)
        },
        "defendant_types": {
            "unique_count": len(all_defendant_types),
            "total_instances": sum(all_defendant_types.values()),
            "similar_groups": find_similar_terms(all_defendant_types),
            "all_terms": dict(all_defendant_types)
        },
        "legal_bases": {
            "unique_count": len(all_legal_bases),
            "total_instances": sum(all_legal_bases.values()),
            "similar_groups": find_similar_terms(all_legal_bases),
            "all_terms": dict(all_legal_bases.most_common(30))
        }
    }
    
    # Save suggestions
    with open(SUGGESTIONS_OUTPUT, "w") as f:
        json.dump(suggestions, f, indent=2)
    
    return suggestions


def categorize_outcome(outcome_text):
    """Categorize outcome into standard categories"""
    outcome_text = outcome_text.lower()
    
    if any(word in outcome_text for word in ['granted', 'summary judgment for defendant', 'dismissed', 'defendant prevail']):
        return 'defendant_win'
    elif any(word in outcome_text for word in ['denied', 'summary judgment denied', 'plaintiff prevail', 'verdict for plaintiff']):
        return 'plaintiff_win'
    elif any(word in outcome_text for word in ['trial', 'jury', 'further examination', 'remand']):
        return 'pending_trial'
    elif any(word in outcome_text for word in ['settled', 'settlement']):
        return 'settled'
    elif any(word in outcome_text for word in ['partial', 'mixed']):
        return 'mixed'
    else:
        return 'other'


def analyze_statistics(cases, mappings):
    """Generate statistics with normalized terms"""
    print("\n" + "=" * 70)
    print("ANALYZING CASE PATTERNS (with term standardization)")
    print("=" * 70)
    
    stats = {
        "total_cases": len(cases),
        "ai_relevance_distribution": Counter(),
        "party_types": {
            "plaintiff": Counter(),
            "defendant": Counter(),
            "combinations": Counter()
        },
        "claims": {
            "by_type": defaultdict(lambda: {"count": 0, "cases": [], "outcomes": Counter()}),
            "by_legal_basis": defaultdict(lambda: {"count": 0, "cases": [], "outcomes": Counter()}),
            "total_claims": 0
        },
        "defenses": {
            "by_type": defaultdict(lambda: {"count": 0, "cases": [], "legal_bases": Counter()}),
            "total_defenses": 0
        },
        "outcomes": {
            "overall": Counter(),
            "by_claim_type": defaultdict(Counter),
            "by_legal_basis": defaultdict(Counter)
        },
        "claim_defense_pairs": Counter(),
        "multi_claim_analysis": {
            "single_claim_cases": 0,
            "multi_claim_cases": 0,
            "avg_claims_per_case": 0,
            "max_claims_in_case": 0
        }
    }
    
    claim_counts = []
    
    for case in cases:
        case_id = case.get("case_id", "unknown")
        
        # AI relevance
        stats["ai_relevance_distribution"][case.get("ai_relevance", "UNKNOWN")] += 1
        
        # Party types (with mapping)
        parties = case.get("parties", {})
        plaintiff_type = apply_mappings(parties.get("plaintiff_type", "Unknown"), mappings)
        defendant_type = apply_mappings(parties.get("defendant_type", "Unknown"), mappings)
        
        stats["party_types"]["plaintiff"][plaintiff_type] += 1
        stats["party_types"]["defendant"][defendant_type] += 1
        stats["party_types"]["combinations"][f"{plaintiff_type} v. {defendant_type}"] += 1
        
        # Claims
        claims = case.get("claims", [])
        claim_counts.append(len(claims))
        
        if len(claims) == 1:
            stats["multi_claim_analysis"]["single_claim_cases"] += 1
        elif len(claims) > 1:
            stats["multi_claim_analysis"]["multi_claim_cases"] += 1
        
        stats["multi_claim_analysis"]["max_claims_in_case"] = max(
            stats["multi_claim_analysis"]["max_claims_in_case"],
            len(claims)
        )
        
        case_claim_types = []
        
        for claim in claims:
            stats["claims"]["total_claims"] += 1
            
            # Extract and normalize claim type
            claim_text = claim.get("claim", "")
            claim_type = apply_mappings(extract_claim_type(claim_text), mappings)
            case_claim_types.append(claim_type)
            
            stats["claims"]["by_type"][claim_type]["count"] += 1
            if case_id not in stats["claims"]["by_type"][claim_type]["cases"]:
                stats["claims"]["by_type"][claim_type]["cases"].append(case_id)
            
            # Legal basis
            legal_basis = claim.get("legal_basis", "")
            legal_basis_type = apply_mappings(extract_legal_basis_type(legal_basis), mappings)
            
            stats["claims"]["by_legal_basis"][legal_basis_type]["count"] += 1
            if case_id not in stats["claims"]["by_legal_basis"][legal_basis_type]["cases"]:
                stats["claims"]["by_legal_basis"][legal_basis_type]["cases"].append(case_id)
            
            # Outcomes
            outcome = normalize_text(claim.get("outcome", ""))
            outcome_category = categorize_outcome(outcome)
            
            stats["outcomes"]["overall"][outcome_category] += 1
            stats["claims"]["by_type"][claim_type]["outcomes"][outcome_category] += 1
            stats["claims"]["by_legal_basis"][legal_basis_type]["outcomes"][outcome_category] += 1
            stats["outcomes"]["by_claim_type"][claim_type][outcome_category] += 1
            stats["outcomes"]["by_legal_basis"][legal_basis_type][outcome_category] += 1
        
        # Defenses
        defenses = case.get("defenses", [])
        
        for defense in defenses:
            stats["defenses"]["total_defenses"] += 1
            
            defense_text = defense.get("defense", "")
            defense_type = apply_mappings(extract_claim_type(defense_text), mappings)
            
            stats["defenses"]["by_type"][defense_type]["count"] += 1
            if case_id not in stats["defenses"]["by_type"][defense_type]["cases"]:
                stats["defenses"]["by_type"][defense_type]["cases"].append(case_id)
            
            legal_basis = defense.get("legal_basis", "")
            if legal_basis:
                stats["defenses"]["by_type"][defense_type]["legal_bases"][legal_basis] += 1
            
            # Claim-defense pairs
            for claim_type in case_claim_types:
                stats["claim_defense_pairs"][(claim_type, defense_type)] += 1
    
    # Calculate averages
    if claim_counts:
        stats["multi_claim_analysis"]["avg_claims_per_case"] = sum(claim_counts) / len(claim_counts)
    
    # Convert to regular dicts
    stats["claims"]["by_type"] = dict(stats["claims"]["by_type"])
    stats["claims"]["by_legal_basis"] = dict(stats["claims"]["by_legal_basis"])
    stats["defenses"]["by_type"] = dict(stats["defenses"]["by_type"])
    stats["outcomes"]["by_claim_type"] = dict(stats["outcomes"]["by_claim_type"])
    stats["outcomes"]["by_legal_basis"] = dict(stats["outcomes"]["by_legal_basis"])
    stats["ai_relevance_distribution"] = dict(stats["ai_relevance_distribution"])
    stats["party_types"]["plaintiff"] = dict(stats["party_types"]["plaintiff"])
    stats["party_types"]["defendant"] = dict(stats["party_types"]["defendant"])
    stats["party_types"]["combinations"] = dict(stats["party_types"]["combinations"])
    stats["outcomes"]["overall"] = dict(stats["outcomes"]["overall"])
    
    stats["claim_defense_pairs"] = [
        {"claim": pair[0], "defense": pair[1], "frequency": count}
        for pair, count in stats["claim_defense_pairs"].most_common(25)
    ]
    
    print(f"✓ Analyzed {stats['total_cases']} cases")
    print(f"✓ Found {stats['claims']['total_claims']} total claims")
    print(f"✓ Found {stats['defenses']['total_defenses']} total defenses")
    
    return stats


def calculate_success_rate(outcome_counter):
    """Calculate plaintiff success rate"""
    total = sum(outcome_counter.values())
    if total == 0:
        return 0.0
    
    wins = outcome_counter.get("plaintiff_win", 0)
    mixed = outcome_counter.get("mixed", 0) * 0.5
    
    return round(((wins + mixed) / total) * 100, 1)


def generate_detailed_breakdowns(stats):
    """Generate detailed breakdowns"""
    detailed = {
        "top_claims_ranked": [],
        "top_defenses_ranked": [],
        "top_legal_bases_ranked": [],
        "outcome_success_rates": {},
    }
    
    # Top claims
    sorted_claims = sorted(
        stats["claims"]["by_type"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    for claim_type, data in sorted_claims[:20]:
        detailed["top_claims_ranked"].append({
            "claim_type": claim_type,
            "frequency": data["count"],
            "case_count": len(data["cases"]),
            "outcomes": dict(data["outcomes"]),
            "success_rate": calculate_success_rate(data["outcomes"])
        })
    
    # Top defenses
    sorted_defenses = sorted(
        stats["defenses"]["by_type"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    for defense_type, data in sorted_defenses[:20]:
        detailed["top_defenses_ranked"].append({
            "defense_type": defense_type,
            "frequency": data["count"],
            "case_count": len(data["cases"]),
            "legal_bases": dict(data["legal_bases"])
        })
    
    # Top legal bases
    sorted_legal_bases = sorted(
        stats["claims"]["by_legal_basis"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    for legal_basis, data in sorted_legal_bases[:15]:
        detailed["top_legal_bases_ranked"].append({
            "legal_basis": legal_basis,
            "frequency": data["count"],
            "case_count": len(data["cases"]),
            "outcomes": dict(data["outcomes"]),
            "success_rate": calculate_success_rate(data["outcomes"])
        })
    
    # Success rates
    total_outcomes = sum(stats["outcomes"]["overall"].values())
    if total_outcomes > 0:
        plaintiff_rate = calculate_success_rate(stats["outcomes"]["overall"])
        defendant_wins = stats["outcomes"]["overall"].get("defendant_win", 0)
        defendant_rate = round((defendant_wins / total_outcomes) * 100, 1)
    else:
        plaintiff_rate = 0.0
        defendant_rate = 0.0
    
    detailed["outcome_success_rates"] = {
        "plaintiff": plaintiff_rate,
        "defendant": defendant_rate,
        "by_outcome": stats["outcomes"]["overall"]
    }
    
    return detailed


def generate_report(stats, detailed):
    """Generate human-readable report"""
    lines = []
    
    lines.append("=" * 70)
    lines.append("LEGAL CASE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"\nTotal Cases: {stats['total_cases']}")
    lines.append(f"Total Claims: {stats['claims']['total_claims']}")
    lines.append(f"Total Defenses: {stats['defenses']['total_defenses']}")
    lines.append(f"Avg Claims/Case: {stats['multi_claim_analysis']['avg_claims_per_case']:.2f}")
    
    # AI Relevance
    lines.append("\n" + "=" * 70)
    lines.append("AI RELEVANCE")
    lines.append("=" * 70)
    for relevance, count in sorted(stats["ai_relevance_distribution"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['total_cases']) * 100
        lines.append(f"{relevance}: {count} ({pct:.1f}%)")
    
    # Parties
    lines.append("\n" + "=" * 70)
    lines.append("PARTY TYPES")
    lines.append("=" * 70)
    
    lines.append("\nPlaintiffs:")
    for party, count in sorted(stats["party_types"]["plaintiff"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['total_cases']) * 100
        lines.append(f"  {party}: {count} ({pct:.1f}%)")
    
    lines.append("\nDefendants:")
    for party, count in sorted(stats["party_types"]["defendant"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['total_cases']) * 100
        lines.append(f"  {party}: {count} ({pct:.1f}%)")
    
    # Top claims
    lines.append("\n" + "=" * 70)
    lines.append("TOP 20 CLAIMS")
    lines.append("=" * 70)
    
    for idx, claim in enumerate(detailed["top_claims_ranked"][:20], 1):
        lines.append(f"\n{idx}. {claim['claim_type'].upper()}")
        lines.append(f"   Count: {claim['frequency']} | Cases: {claim['case_count']} | Success: {claim['success_rate']}%")
        lines.append(f"   Outcomes: {claim['outcomes']}")
    
    # Top defenses
    lines.append("\n" + "=" * 70)
    lines.append("TOP 20 DEFENSES")
    lines.append("=" * 70)
    
    for idx, defense in enumerate(detailed["top_defenses_ranked"][:20], 1):
        lines.append(f"\n{idx}. {defense['defense_type'].upper()}")
        lines.append(f"   Count: {defense['frequency']} | Cases: {defense['case_count']}")
    
    # Legal bases
    lines.append("\n" + "=" * 70)
    lines.append("TOP LEGAL BASES")
    lines.append("=" * 70)
    
    for idx, basis in enumerate(detailed["top_legal_bases_ranked"][:15], 1):
        lines.append(f"\n{idx}. {basis['legal_basis'].upper()}")
        lines.append(f"   Count: {basis['frequency']} | Success: {basis['success_rate']}%")
    
    # Claim-defense pairs
    lines.append("\n" + "=" * 70)
    lines.append("CLAIM-DEFENSE PAIRS")
    lines.append("=" * 70)
    
    for idx, pair in enumerate(stats["claim_defense_pairs"][:15], 1):
        lines.append(f"\n{idx}. [{pair['frequency']}x] {pair['claim']} → {pair['defense']}")
    
    # Outcomes
    lines.append("\n" + "=" * 70)
    lines.append("OUTCOMES")
    lines.append("=" * 70)
    lines.append(f"\nPlaintiff Success: {detailed['outcome_success_rates']['plaintiff']}%")
    lines.append(f"Defendant Success: {detailed['outcome_success_rates']['defendant']}%")
    
    lines.append("\nBreakdown:")
    for outcome, count in sorted(stats["outcomes"]["overall"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats['claims']['total_claims']) * 100
        lines.append(f"  {outcome}: {count} ({pct:.1f}%)")
    
    report_text = "\n".join(lines)
    
    return report_text


def main():
    """Main execution"""
    print("\n🏛️  ENHANCED LEGAL CASE ANALYZER")
    print("=" * 70)
    
    # Load data
    print("\nLoading cases...")
    with open(INPUT_PATH, "r") as f:
        cases = json.load(f)
    print(f"✓ Loaded {len(cases)} cases")
    
    # Step 1: Analyze and suggest standardizations
    suggestions = analyze_and_suggest_standardization(cases)
    
    # Step 2: Load or create mappings
    mappings = load_or_create_mappings()
    
    # Step 3: Run analysis with standardized terms
    stats = analyze_statistics(cases, mappings)
    
    # Step 4: Generate detailed breakdowns
    detailed = generate_detailed_breakdowns(stats)
    
    # Step 5: Save outputs
    raw_output = {
        "statistics": stats,
        "detailed": detailed,
        "suggestions": suggestions
    }

    with open(RAW_OUTPUT_PATH, "w") as f:
        json.dump(raw_output, f, indent=2)

    # Step 6: Generate report
    generate_report(stats, detailed)

    if suggestions['claim_types']['similar_groups'] or suggestions['plaintiff_types']['similar_groups']:
        print(f"\n💡 TIP: Review {SUGGESTIONS_OUTPUT} for term standardization opportunities")
        print(f"   Then edit {MAPPINGS_PATH} and re-run for cleaner results")


if __name__ == "__main__":
    main()