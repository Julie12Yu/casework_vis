import json

# Privacy and Data Protection
# IP Law
# Consumer Protection
NAME = 'Privacy and Data Protection'

def filter_privacy_cases(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cases = data.get('documents', [])

    filtered_cases = []
    for case in cases:
        if isinstance(case, dict):
            if case.get('legal_category_name') == NAME:
                filtered_cases.append(case)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_cases, f, indent=2, ensure_ascii=False)

    print(f"Total cases found in category '{NAME}': {len(filtered_cases)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Specify your input and output file paths
    input_file = "../misc/new_court_cases_processed.json"
    output_file = f"{NAME}.json"
    
    filter_privacy_cases(input_file, output_file)




"""
{
    "case_id": "2024-07-22_USCOURTS-flsd-1_23-cv-23306_MAZILE v. Larkin University Corp. et al.pdf",
    "case_name": "Mazile v. Larkin University Corp. and ExamSoft Worldwide, Inc.",
    "ai_relevance": "AI-based remote proctoring flagged plaintiff as cheating, triggering discipline.",
    "parties": {
      "plaintiff_name": "Christelle Mazile",
      "plaintiff_description": "Individual doctoral student at Larkin University",
      "defendant_name": "Larkin University Corp. and ExamSoft",
      "defendant_description": "Private university and exam software company"
    },
    "claims": [
      {
        "claim": "Disability discrimination under the Americans with Disabilities Act (ADA) against Larkin University",
        "legal_basis": "Title III of the Americans with Disabilities Act, 42 U.S.C. \u00a7 12181 et seq.",
        "outcome": "Dismissed without prejudice for failure to plausibly allege discriminatory conduct or knowledge by Larkin regarding ExamSoft\u2019s alleged discriminatory effects."
      },
      {
        "claim": "Disability discrimination under the Rehabilitation Act (RA) against Larkin University",
        "legal_basis": "Section 504 of the Rehabilitation Act of 1973, 29 U.S.C. \u00a7 794",
        "outcome": "Dismissed without prejudice for insufficient factual allegations that Larkin, as a funding recipient, discriminated against plaintiff because of disability or knew of ExamSoft\u2019s alleged discriminatory impact."
      },
      {
        "claim": "Constitutional rights violation under 42 U.S.C. \u00a7 1983 against Larkin University",
        "legal_basis": "42 U.S.C. \u00a7 1983 (requiring state action)",
        "outcome": "Dismissed with prejudice because Larkin University is a private institution and not a state actor, and no plausible state-action theory was alleged."
      },
      {
        "claim": "Other statutory or common-law discrimination / wrongful expulsion claims against Larkin University",
        "legal_basis": "Various federal and/or state anti-discrimination and related statutes (not fully specified in summary)",
        "outcome": "Dismissed without prejudice for failure to allege facts showing wrongful discrimination or that Larkin knew of and adopted any discriminatory effects of ExamSoft\u2019s software."
      },
      {
        "claim": "Claims against ExamSoft arising from use of its AI-proctored exam software (e.g., negligence, product liability, breach of contract, or related statutory claims)",
        "legal_basis": "State contract and/or tort law and related statutory claims governed by ExamSoft\u2019s End User License Agreement (EULA)",
        "outcome": "Not decided on the merits; court compelled arbitration and stayed/removed them from judicial forum based on enforceable arbitration clause in clickwrap EULA."
      }
    ],
    "defenses": [
      {
        "defense": "Enforceable arbitration agreement in ExamSoft\u2019s EULA requires all claims against ExamSoft to be arbitrated, not litigated in court.",
        "legal_basis": "Federal Arbitration Act and contract law principles enforcing clickwrap arbitration clauses."
      },
      {
        "defense": "Larkin University is not a state actor and therefore cannot be sued under 42 U.S.C. \u00a7 1983.",
        "legal_basis": "State-action requirement under \u00a7 1983 and related constitutional jurisprudence."
      },
      {
        "defense": "Plaintiff failed to plausibly allege that Larkin intentionally discriminated against her or had knowledge of any discriminatory impact of ExamSoft\u2019s AI-proctoring on her disabilities.",
        "legal_basis": "Pleading standards under Federal Rule of Civil Procedure 12(b)(6) and substantive elements of ADA and RA discrimination claims."
      },
      {
        "defense": "Use of third-party exam software and reliance on its results, without more, does not establish discriminatory animus or deliberate indifference by Larkin.",
        "legal_basis": "Substantive standards for disability discrimination and failure-to-accommodate claims under ADA/RA and related case law."
      }
    ]
  },

"""