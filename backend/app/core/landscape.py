"""First-turn landscape brief: aggregate the corpus by population/phase/status.

Computed deterministically from the in-memory document list, not from top-k
retrieval, so the clinician sees the full population before drilling in.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# (regex, display name). Order matters: match more specific populations first.
_DISEASE_PATTERNS: list[tuple[str, str]] = [
    (r"triple[\s\-]*negative|\btnbc\b", "Triple-Negative Breast Cancer (TNBC)"),
    (r"hr[\+\s\-]*positive|hormone[\s\-]*receptor[\s\-]*positive|\ber[\+\s\-]*positive\b|estrogen[\s\-]*receptor[\s\-]*positive", "HR+ Breast Cancer"),
    (r"her2[\s\-]*low\b", "HER2-Low Breast Cancer"),
    (r"her2[\+\s\-]*positive|\bher2\+", "HER2+ Breast Cancer"),
    (r"inflammatory[\s\-]*breast", "Inflammatory Breast Cancer"),
    (r"\bdcis\b|ductal[\s\-]*carcinoma[\s\-]*in[\s\-]*situ", "Ductal Carcinoma in Situ (DCIS)"),
    (r"male[\s\-]*breast", "Male Breast Cancer"),
    (r"lobular[\s\-]*(carcinoma|breast)", "Lobular Breast Cancer"),
    (r"breast[\s\-]*(cancer|carcinoma|neoplasm|tumou?r)", "Breast Cancer (general)"),
]


def detect_disease(query: str) -> Optional[tuple[str, str]]:
    q = query.lower()
    for pattern, name in _DISEASE_PATTERNS:
        if re.search(pattern, q):
            return pattern, name
    return None


def detect_phase(query: str) -> Optional[str]:
    q = query.lower()
    if re.search(r"early[\s\-]*phase[\s\-]*1\b", q):
        return "EARLY_PHASE1"
    for num, label in [("4", "PHASE4"), ("3", "PHASE3"), ("2", "PHASE2"), ("1", "PHASE1")]:
        if re.search(rf"\bphase[\s\-]*{num}\b", q):
            return label
    # Roman: longer alternatives first so "iii" doesn't get partial-matched as "ii".
    for roman, label in [("iv", "PHASE4"), ("iii", "PHASE3"), ("ii", "PHASE2"), ("i", "PHASE1")]:
        if re.search(rf"\bphase[\s\-]*{roman}\b", q):
            return label
    return None


def detect_status(query: str) -> Optional[str]:
    q = query.lower()
    if re.search(r"not\s*yet\s*recruiting", q):
        return "NOT_YET_RECRUITING"
    if re.search(r"active[\s,]*not[\s\-]*recruiting", q):
        return "ACTIVE_NOT_RECRUITING"
    if re.search(r"\b(recruiting|enrolling|actively\s+enrolling)\b", q):
        return "RECRUITING"
    if re.search(r"\bcompleted\b", q):
        return "COMPLETED"
    if re.search(r"\bterminated\b", q):
        return "TERMINATED"
    if re.search(r"\bwithdrawn\b", q):
        return "WITHDRAWN"
    return None


# Drug classes matched against interventionName + interventionOtherNames.
_DRUG_CLASSES: list[tuple[str, str]] = [
    ("PD-(L)1 / immune checkpoint",
     r"pembrolizumab|atezolizumab|nivolumab|durvalumab|avelumab|cemiplimab|tislelizumab|toripalimab|camrelizumab|sintilimab|ipilimumab|tremelimumab|relatlimab|tiragolumab|checkpoint|anti.?pd.?l?1"),
    ("PARP inhibitor",
     r"olaparib|talazoparib|niraparib|rucaparib|veliparib|fluzoparib|pamiparib|\bparp\b"),
    ("ADC (antibody-drug conjugate)",
     r"sacituzumab|trastuzumab\s+deruxtecan|trastuzumab\s+emtansine|trodelvy|enhertu|kadcyla|datopotamab|patritumab\s+deruxtecan|disitamab|ladiratuzumab|mirvetuximab|antibody.?drug\s+conjugate"),
    ("HER2-targeted therapy",
     r"trastuzumab|pertuzumab|tucatinib|lapatinib|neratinib|herceptin|perjeta|nerlynx|tukysa|margetuximab|zanidatamab"),
    ("CDK4/6 inhibitor",
     r"palbociclib|ribociclib|abemaciclib|ibrance|kisqali|verzenio"),
    ("AKT / PI3K / mTOR inhibitor",
     r"capivasertib|ipatasertib|alpelisib|inavolisib|everolimus|temsirolimus|truqap|piqray|afinitor|\bakt\b|\bpi3k\b|\bmtor\b"),
    ("Endocrine / SERD / aromatase",
     r"tamoxifen|fulvestrant|elacestrant|giredestrant|camizestrant|letrozole|anastrozole|exemestane|aromatase|orserdu"),
    ("AR-directed (LAR-TNBC)",
     r"enzalutamide|bicalutamide|abiraterone|apalutamide|darolutamide|androgen\s+receptor"),
    ("VEGF / anti-angiogenic",
     r"bevacizumab|ramucirumab|apatinib|anlotinib|sunitinib|sorafenib|cabozantinib|lenvatinib|axitinib|\bvegf\b|avastin"),
    ("Taxane chemotherapy",
     r"paclitaxel|nab.?paclitaxel|abraxane|docetaxel|cabazitaxel"),
    ("Platinum chemotherapy",
     r"carboplatin|cisplatin|oxaliplatin"),
    ("Anthracycline chemotherapy",
     r"doxorubicin|epirubicin|adriamycin"),
    ("Alkylating chemotherapy",
     r"cyclophosphamide|ifosfamide|temozolomide"),
    ("Capecitabine / 5-FU / gemcitabine",
     r"capecitabine|5-fu|fluorouracil|gemcitabine|xeloda"),
    ("Eribulin",
     r"eribulin|halaven"),
    ("Vaccine / cell therapy / TIL",
     r"\bvaccine\b|car.?t\b|\btils\b|tumor.?infiltrating|dendritic"),
    ("Radiation / radiotherapy",
     r"\bradiation\b|radiotherapy|sbrt|imrt|brachytherapy|proton\s+therapy"),
]


def classify_trial(doc: dict) -> list[str]:
    """Return every drug-class label that matches this trial's intervention names."""
    md = doc.get("metadata", {})
    hay = (
        md.get("interventionName", "") + " || " + md.get("interventionOtherNames", "")
    ).lower()
    return [label for label, pat in _DRUG_CLASSES if re.search(pat, hay)]


_OTHER_CLASS_LABEL = "(other / unclassified)"


def diversify_by_drug_class(retrieved: list[dict], budget: int) -> list[dict]:
    """Pick ~budget trials so each drug class is represented before doubling up.

    `retrieved` is assumed sorted by descending relevance. Round-robin walks the
    classes (ordered by where their top-ranked trial sits in the list) and grabs
    each class's highest-ranked unseen trial. Falls back to "(other)" for trials
    whose interventions don't match any pattern.
    """
    if budget <= 0 or not retrieved:
        return []

    by_class: dict[str, list[int]] = {}
    for i, doc in enumerate(retrieved):
        classes = classify_trial(doc) or [_OTHER_CLASS_LABEL]
        for c in classes:
            by_class.setdefault(c, []).append(i)

    # Order classes by where their top trial sits (so the highest-relevance class
    # is the first round-robin pick).
    class_order = sorted(by_class.keys(), key=lambda c: by_class[c][0])

    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < budget:
        progress = False
        for c in class_order:
            for idx in by_class[c]:
                if idx not in seen:
                    chosen.append(idx)
                    seen.add(idx)
                    progress = True
                    break
            if len(chosen) >= budget:
                break
        if not progress:
            break

    chosen.sort()  # preserve relative retrieval order in the final list
    return [retrieved[i] for i in chosen]


def format_landscape_for_llm(landscape: dict) -> str:
    """Render landscape stats as a compact text block suitable for prompt injection."""
    if not landscape or not landscape.get("total"):
        return ""
    fa = landscape.get("filters_applied", {}) or {}
    lines = [
        f"Population: {landscape.get('population', 'unspecified')}",
        f"Filters applied: phase={fa.get('phase') or 'any'}, status={fa.get('status') or 'any'}",
        f"Total trials matching the population + filters: {landscape['total']}",
    ]
    disease_total = landscape.get("disease_total")
    if disease_total and disease_total != landscape["total"]:
        lines.append(f"(of {disease_total} total {landscape.get('population', '')} trials in the corpus)")

    drug_classes = landscape.get("drug_classes") or []
    if drug_classes:
        lines.append("Drug-class / modality counts in this matching set (approximate, pattern-based):")
        for c in drug_classes[:14]:
            lines.append(f"  - {c['class']}: {c['count']}")

    status_dist = landscape.get("status_distribution") or []
    if status_dist:
        lines.append("Status distribution:")
        for s in status_dist[:8]:
            lines.append(f"  - {s['status']}: {s['count']}")
    return "\n".join(lines)


def _matches_disease(doc: dict, pattern: str) -> bool:
    """Per project convention, infer disease only from title/conditions/keywords/MeSH —
    never from summary/description/inclusion criteria."""
    md = doc.get("metadata", {})
    haystack = " || ".join([
        md.get("title", ""),
        md.get("officialTitle", ""),
        md.get("conditions", ""),
        md.get("keywords", ""),
        md.get("meshTermsCondition", ""),
    ]).lower()
    return bool(re.search(pattern, haystack))


def extract_landscape_filters(query: str) -> dict:
    """Return a filters dict for a single utterance, or {} when no population is detected."""
    disease = detect_disease(query)
    if not disease:
        return {}
    pattern, name = disease
    return {
        "disease_pattern": pattern,
        "disease_name": name,
        "phase": detect_phase(query),
        "status": detect_status(query),
    }


def merge_filters_from_history(messages: list[dict]) -> dict:
    """Walk the user turns in order and carry forward each filter (latest wins).

    Returns {} if no disease has ever been mentioned. Otherwise returns the
    standard filters dict with the most recent value for each field. This is
    how follow-up turns inherit "TNBC" from turn 1 and how a turn-2 phase
    declaration overrides a turn-1 phase.
    """
    disease_pattern = None
    disease_name = None
    phase = None
    status = None
    for m in messages:
        if m.get("role") != "user":
            continue
        text = m.get("content", "") or ""
        d = detect_disease(text)
        if d:
            disease_pattern, disease_name = d
        p = detect_phase(text)
        if p:
            phase = p
        s = detect_status(text)
        if s:
            status = s
    if not disease_pattern:
        return {}
    return {
        "disease_pattern": disease_pattern,
        "disease_name": disease_name,
        "phase": phase,
        "status": status,
    }


def compute_landscape(documents: list[dict], filters: dict) -> dict:
    """Aggregate the document corpus under the given filters.

    Returns a dict shaped for the SSE `landscape` event. `total == 0` is a valid
    result the frontend should still render (so the clinician sees the filters
    were too strict)."""
    if not filters:
        return {}

    pattern = filters["disease_pattern"]
    target_phase = filters.get("phase")
    target_status = filters.get("status")

    pool = [d for d in documents if _matches_disease(d, pattern)]
    disease_total = len(pool)

    if target_phase:
        pool = [d for d in pool if d.get("metadata", {}).get("phases", "").strip() == target_phase]
    if target_status:
        pool = [d for d in pool if d.get("metadata", {}).get("status", "").strip() == target_status]

    n = len(pool)

    base = {
        "population": filters["disease_name"],
        "filters_applied": {"phase": target_phase, "status": target_status},
        "total": n,
        "disease_total": disease_total,
        "phase_distribution": [],
        "status_distribution": [],
        "intervention_types": [],
        "drug_classes": [],
        "geography_top": [],
        "sponsor_class": [],
    }
    if n == 0:
        return base

    phase_counts = Counter(
        (d.get("metadata", {}).get("phases", "").strip() or "(unspecified)") for d in pool
    )
    status_counts = Counter(
        (d.get("metadata", {}).get("status", "").strip() or "(unspecified)") for d in pool
    )

    int_type_counts: Counter = Counter()
    for d in pool:
        for t in d.get("metadata", {}).get("interventionType", "").split(","):
            t = t.strip().upper()
            if t:
                int_type_counts[t] += 1

    drug_class_counts: list[tuple[str, int]] = []
    for label, drug_pat in _DRUG_CLASSES:
        cnt = 0
        for d in pool:
            md = d.get("metadata", {})
            hay = (md.get("interventionName", "") + " || " + md.get("interventionOtherNames", "")).lower()
            if re.search(drug_pat, hay):
                cnt += 1
        if cnt > 0:
            drug_class_counts.append((label, cnt))
    drug_class_counts.sort(key=lambda x: -x[1])

    country_counts: Counter = Counter()
    for d in pool:
        for c in re.split(r"[,;|]", d.get("metadata", {}).get("locationCountries", "")):
            c = c.strip()
            if c:
                country_counts[c] += 1

    sponsor_counts = Counter(
        (d.get("metadata", {}).get("sponsorClass", "").strip() or "(unspecified)") for d in pool
    )

    base.update({
        "phase_distribution": [{"phase": k, "count": v} for k, v in phase_counts.most_common()],
        "status_distribution": [{"status": k, "count": v} for k, v in status_counts.most_common()],
        "intervention_types": [{"type": k, "count": v} for k, v in int_type_counts.most_common()],
        "drug_classes": [{"class": k, "count": v} for k, v in drug_class_counts[:20]],
        "geography_top": [{"country": k, "count": v} for k, v in country_counts.most_common(10)],
        "sponsor_class": [{"class": k, "count": v} for k, v in sponsor_counts.most_common()],
    })
    return base
