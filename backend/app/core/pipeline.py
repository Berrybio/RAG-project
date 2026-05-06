import logging
import re

from .data import build_documents, load_clinical_trials
from .feedback_reranker import adjust_scores
from .generation import generate_answer, generate_answer_stream
from .landscape import classify_trial
from .llm import BaseLLMProvider
from .retriever import TFIDFRetriever, VoyageRetriever

logger = logging.getLogger(__name__)

# Country aliases → canonical name used in locationCountries field.
# Keys are matched case-insensitively against the query.
_COUNTRY_ALIASES: dict[str, str] = {
    "united states": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "america": "United States",
    "american": "United States",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "china": "China",
    "chinese": "China",
    "canada": "Canada",
    "canadian": "Canada",
    "france": "France",
    "french": "France",
    "germany": "Germany",
    "german": "Germany",
    "japan": "Japan",
    "japanese": "Japan",
    "south korea": "South Korea",
    "korea": "South Korea",
    "korean": "South Korea",
    "australia": "Australia",
    "australian": "Australia",
    "italy": "Italy",
    "italian": "Italy",
    "spain": "Spain",
    "spanish": "Spain",
    "netherlands": "Netherlands",
    "dutch": "Netherlands",
    "belgium": "Belgium",
    "switzerland": "Switzerland",
    "swiss": "Switzerland",
    "india": "India",
    "indian": "India",
    "brazil": "Brazil",
    "brazilian": "Brazil",
    "mexico": "Mexico",
    "mexican": "Mexico",
    "turkey": "Turkey (Türkiye)",
    "turkish": "Turkey (Türkiye)",
}


def _extract_countries(query: str) -> list[str]:
    """Return canonical country names mentioned in the query, longest-alias-first."""
    q = query.lower()
    found: list[str] = []
    # Sort aliases by length desc so "united states" matches before "us".
    for alias in sorted(_COUNTRY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            canonical = _COUNTRY_ALIASES[alias]
            if canonical not in found:
                found.append(canonical)
            # Remove matched span so shorter aliases ("us") don't double-match inside "united states".
            q = re.sub(rf"\b{re.escape(alias)}\b", " ", q)
    return found


def _doc_matches_countries(doc: dict, countries: list[str]) -> bool:
    """True if any of the target countries appears in the doc's locationCountries."""
    haystack = (doc.get("metadata", {}).get("locationCountries") or "").lower()
    return any(c.lower() in haystack for c in countries)


# ---- US state / sub-national region filter ----
#
# Matches queries like "trials in New York" or "Texas trials" against the
# full per-trial location string, so multi-site trials that include the
# requested state anywhere in their locations are kept.

_US_STATES: dict[str, str] = {
    # full name : canonical form used for substring match (title-case)
    "alabama": "Alabama", "alaska": "Alaska", "arizona": "Arizona", "arkansas": "Arkansas",
    "california": "California", "colorado": "Colorado", "connecticut": "Connecticut",
    "delaware": "Delaware", "florida": "Florida", "georgia": "Georgia", "hawaii": "Hawaii",
    "idaho": "Idaho", "illinois": "Illinois", "indiana": "Indiana", "iowa": "Iowa",
    "kansas": "Kansas", "kentucky": "Kentucky", "louisiana": "Louisiana", "maine": "Maine",
    "maryland": "Maryland", "massachusetts": "Massachusetts", "michigan": "Michigan",
    "minnesota": "Minnesota", "mississippi": "Mississippi", "missouri": "Missouri",
    "montana": "Montana", "nebraska": "Nebraska", "nevada": "Nevada",
    "new hampshire": "New Hampshire", "new jersey": "New Jersey", "new mexico": "New Mexico",
    "new york": "New York", "north carolina": "North Carolina", "north dakota": "North Dakota",
    "ohio": "Ohio", "oklahoma": "Oklahoma", "oregon": "Oregon", "pennsylvania": "Pennsylvania",
    "rhode island": "Rhode Island", "south carolina": "South Carolina",
    "south dakota": "South Dakota", "tennessee": "Tennessee", "texas": "Texas", "utah": "Utah",
    "vermont": "Vermont", "virginia": "Virginia", "washington": "Washington",
    "west virginia": "West Virginia", "wisconsin": "Wisconsin", "wyoming": "Wyoming",
    "district of columbia": "District of Columbia", "washington dc": "District of Columbia",
    "washington, d.c.": "District of Columbia",
}


def _extract_us_states(query: str) -> list[str]:
    """Return canonical US state names referenced in the query."""
    q = query.lower()
    found: list[str] = []
    # Longest alias first so "new york" wins over nothing, and "west virginia" over "virginia".
    for alias in sorted(_US_STATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            canonical = _US_STATES[alias]
            if canonical not in found:
                found.append(canonical)
            q = re.sub(rf"\b{re.escape(alias)}\b", " ", q)
    return found


def _doc_matches_states(doc: dict, states: list[str]) -> bool:
    """True if any requested state appears in the doc's full location string.

    Uses the untruncated `locationInfoFull` so multi-site trials with long
    location lists still match sites past the 500-char display truncation.
    """
    meta = doc.get("metadata", {})
    haystack = (meta.get("locationInfoFull") or meta.get("locationInfo") or "").lower()
    return any(s.lower() in haystack for s in states)


# ---- Structured filters: studyType and recruitment status ----
#
# The dense retriever ranks by semantic similarity, so a query like
# "observational trials currently recruiting" — which has weak clinical
# content — often returns mostly irrelevant top-k. Extract these
# attributes from the query and hard-filter against metadata.

def _extract_study_type(query: str) -> str | None:
    """Return canonical studyType if the query names one, else None."""
    q = query.lower()
    if re.search(r"\bobservational\b", q):
        return "OBSERVATIONAL"
    if re.search(r"\binterventional\b", q):
        return "INTERVENTIONAL"
    if re.search(r"\bexpanded[\s-]access\b", q):
        return "EXPANDED_ACCESS"
    return None


# Map natural-language phrases to canonical oStatus values from ClinicalTrials.gov.
# Order matters: longer/more-specific phrases are matched first.
_STATUS_PATTERNS: list[tuple[str, str]] = [
    (r"\bnot\s+yet\s+recruiting\b", "NOT_YET_RECRUITING"),
    (r"\bactive\s*,?\s*not\s+recruiting\b", "ACTIVE_NOT_RECRUITING"),
    (r"\benrolling\s+by\s+invitation\b", "ENROLLING_BY_INVITATION"),
    (r"\b(currently\s+|actively\s+)?recruiting\b", "RECRUITING"),
    (r"\benrolling\b", "RECRUITING"),  # colloquial synonym
    (r"\bcompleted\b", "COMPLETED"),
    (r"\bterminated\b", "TERMINATED"),
    (r"\bwithdrawn\b", "WITHDRAWN"),
    (r"\bsuspended\b", "SUSPENDED"),
]


def _extract_statuses(query: str) -> list[str]:
    """Return canonical oStatus values implied by the query."""
    q = query.lower()
    found: list[str] = []
    for pattern, canonical in _STATUS_PATTERNS:
        if re.search(pattern, q):
            if canonical not in found:
                found.append(canonical)
            q = re.sub(pattern, " ", q)
    return found


def _doc_matches_study_type(doc: dict, study_type: str) -> bool:
    return (doc.get("metadata", {}).get("studyType") or "").upper() == study_type


def _doc_matches_statuses(doc: dict, statuses: list[str]) -> bool:
    value = (doc.get("metadata", {}).get("status") or "").upper()
    return value in statuses


# ---- Sponsor filter ----
#
# Dense retrieval doesn't rank reliably on sponsor identity, so a query like
# "AstraZeneca's ADC trials for TNBC" can return academic / Chinese-university
# sponsored trials whose text matches "ADC trials for TNBC" more strongly. We
# detect named pharma sponsors in the query and post-filter retrieved docs
# against `sponsorName`.
#
# The dict maps natural-language aliases (lowercased, what users type) to a
# substring that should appear in the doc's sponsorName field. Substring
# match — not equality — so "astrazeneca" matches both "AstraZeneca" and any
# regional sub-entity (e.g. "AstraZeneca AB"). One canonical group per
# alias; "msd" and "merck" both collapse to "merck" so either spelling
# pulls Merck-sponsored trials.

_SPONSOR_ALIASES: dict[str, str] = {
    "astrazeneca": "astrazeneca",
    "merck": "merck",
    "msd": "merck",
    "pfizer": "pfizer",
    "roche": "roche",
    "hoffmann-la roche": "roche",
    "genentech": "genentech",
    "novartis": "novartis",
    "eli lilly": "lilly",
    "lilly": "lilly",
    "janssen": "janssen",
    "johnson & johnson": "janssen",
    "j&j": "janssen",
    "gsk": "glaxosmithkline",
    "glaxosmithkline": "glaxosmithkline",
    "gilead": "gilead",
    "bristol-myers squibb": "bristol",
    "bristol myers squibb": "bristol",
    "bms": "bristol",
    "daiichi sankyo": "daiichi sankyo",
    "daiichi": "daiichi sankyo",
    "sanofi": "sanofi",
    "bayer": "bayer",
    "takeda": "takeda",
    "abbvie": "abbvie",
    "amgen": "amgen",
    "regeneron": "regeneron",
    "seagen": "seagen",
    "seattle genetics": "seagen",
    "incyte": "incyte",
    "moderna": "moderna",
    "biontech": "biontech",
}


def _extract_sponsors(query: str) -> list[str]:
    """Return canonical sponsor substrings implied by the query.

    Uses whole-word match on aliases, longest-first, so 'bristol myers squibb'
    wins over 'bristol' inside it.
    """
    q = query.lower()
    found: list[str] = []
    for alias in sorted(_SPONSOR_ALIASES, key=len, reverse=True):
        # `\b` doesn't match well around symbols (& and -). For aliases that
        # contain those, we fall back to plain substring on the lowered query.
        if re.search(rf"[a-z0-9_]", alias) is None:
            continue
        if any(c in alias for c in "&-"):
            if alias in q:
                canonical = _SPONSOR_ALIASES[alias]
                if canonical not in found:
                    found.append(canonical)
                q = q.replace(alias, " ")
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q):
            canonical = _SPONSOR_ALIASES[alias]
            if canonical not in found:
                found.append(canonical)
            q = re.sub(rf"\b{re.escape(alias)}\b", " ", q)
    return found


def _doc_matches_sponsors(doc: dict, sponsors: list[str]) -> bool:
    """True if doc's sponsorName contains any of the requested substrings."""
    haystack = (doc.get("metadata", {}).get("sponsorName") or "").lower()
    return any(s in haystack for s in sponsors)


# ---- Drug-class filter ----
#
# Queries like "ADC + checkpoint inhibitor combinations" name drug *categories*
# rather than specific drugs. Voyage embeddings rank trials whose text uses
# the category word verbatim (e.g. Chinese-sponsored studies that say "ADC")
# higher than trials that only mention the underlying drug names (Dato-DXd,
# Durvalumab). This systematically hides the major-pharma trials that *are*
# the canonical examples of those classes.
#
# Two-pronged fix when the query mentions a drug class:
#   1. Expand the retrieval query with representative drug names from that
#      class so dense retrieval finds them.
#   2. Post-filter the retrieved pool to require trials to have ALL the
#      requested classes (so "ADC + checkpoint" returns trials that have
#      BOTH, not just one). Falls back to ANY-match if intersection is empty.

# Aliases the user might type in a query → the canonical class label used by
# `classify_trial` (defined in landscape.py). Match is whole-token / lenient
# on punctuation, longest-first.
_DRUG_CLASS_QUERY_ALIASES: dict[str, str] = {
    # ADC
    "adc": "ADC (antibody-drug conjugate)",
    "adcs": "ADC (antibody-drug conjugate)",
    "antibody-drug conjugate": "ADC (antibody-drug conjugate)",
    "antibody drug conjugate": "ADC (antibody-drug conjugate)",
    "antibody-drug conjugates": "ADC (antibody-drug conjugate)",
    "antibody drug conjugates": "ADC (antibody-drug conjugate)",
    # PD-(L)1 / checkpoint
    "checkpoint inhibitor": "PD-(L)1 / immune checkpoint",
    "checkpoint inhibitors": "PD-(L)1 / immune checkpoint",
    "immune checkpoint": "PD-(L)1 / immune checkpoint",
    "immune checkpoint inhibitor": "PD-(L)1 / immune checkpoint",
    "ici": "PD-(L)1 / immune checkpoint",
    "pd-1 inhibitor": "PD-(L)1 / immune checkpoint",
    "pd-l1 inhibitor": "PD-(L)1 / immune checkpoint",
    "pd1 inhibitor": "PD-(L)1 / immune checkpoint",
    "pdl1 inhibitor": "PD-(L)1 / immune checkpoint",
    "pd-1": "PD-(L)1 / immune checkpoint",
    "pd-l1": "PD-(L)1 / immune checkpoint",
    "anti-pd-1": "PD-(L)1 / immune checkpoint",
    "anti-pd-l1": "PD-(L)1 / immune checkpoint",
    # PARP
    "parp inhibitor": "PARP inhibitor",
    "parp inhibitors": "PARP inhibitor",
    "parpi": "PARP inhibitor",
    # CDK4/6
    "cdk4/6 inhibitor": "CDK4/6 inhibitor",
    "cdk4/6": "CDK4/6 inhibitor",
    "cdk inhibitor": "CDK4/6 inhibitor",
    # HER2
    "her2-targeted therapy": "HER2-targeted therapy",
    "her2-targeted": "HER2-targeted therapy",
    "her2 targeted": "HER2-targeted therapy",
    "anti-her2": "HER2-targeted therapy",
    # AKT/PI3K/mTOR
    "pi3k inhibitor": "AKT / PI3K / mTOR inhibitor",
    "akt inhibitor": "AKT / PI3K / mTOR inhibitor",
    "mtor inhibitor": "AKT / PI3K / mTOR inhibitor",
    # SERD / endocrine
    "serd": "Endocrine / SERD / aromatase",
    "endocrine therapy": "Endocrine / SERD / aromatase",
    "aromatase inhibitor": "Endocrine / SERD / aromatase",
    # AR
    "ar inhibitor": "AR-directed (LAR-TNBC)",
    "androgen receptor inhibitor": "AR-directed (LAR-TNBC)",
    # VEGF / anti-angiogenic
    "vegf inhibitor": "VEGF / anti-angiogenic",
    "anti-angiogenic": "VEGF / anti-angiogenic",
    "anti angiogenic": "VEGF / anti-angiogenic",
    # Chemo categories
    "taxane": "Taxane chemotherapy",
    "taxanes": "Taxane chemotherapy",
    "platinum": "Platinum chemotherapy",
    "platinum chemotherapy": "Platinum chemotherapy",
    "anthracycline": "Anthracycline chemotherapy",
    "anthracyclines": "Anthracycline chemotherapy",
}

# Synonyms used to expand the retrieval query when a class is detected.
# Picked to be the highest-recognition specific drug names in the corpus —
# enough to give Voyage embeddings something concrete to anchor on without
# diluting the query into incoherence. Limit per class is small.
_DRUG_CLASS_EXPANSION: dict[str, list[str]] = {
    "ADC (antibody-drug conjugate)": [
        "Datopotamab deruxtecan", "Trastuzumab deruxtecan",
        "Sacituzumab govitecan", "Dato-DXd", "T-DXd",
    ],
    "PD-(L)1 / immune checkpoint": [
        "Pembrolizumab", "Atezolizumab", "Durvalumab", "Nivolumab",
    ],
    "PARP inhibitor": ["Olaparib", "Talazoparib", "Niraparib"],
    "CDK4/6 inhibitor": ["Palbociclib", "Ribociclib", "Abemaciclib"],
    "HER2-targeted therapy": ["Trastuzumab", "Pertuzumab", "Tucatinib"],
    "AKT / PI3K / mTOR inhibitor": ["Capivasertib", "Alpelisib", "Inavolisib"],
    "Endocrine / SERD / aromatase": ["Fulvestrant", "Elacestrant", "Letrozole"],
    "AR-directed (LAR-TNBC)": ["Enzalutamide", "Bicalutamide"],
    "VEGF / anti-angiogenic": ["Bevacizumab", "Apatinib", "Anlotinib"],
    "Taxane chemotherapy": ["Paclitaxel", "Nab-paclitaxel", "Docetaxel"],
    "Platinum chemotherapy": ["Carboplatin", "Cisplatin"],
    "Anthracycline chemotherapy": ["Doxorubicin", "Epirubicin"],
}


def _extract_drug_classes(query: str) -> list[str]:
    """Return canonical drug-class labels mentioned in the query.

    Handles aliases with `/`, `-`, and other punctuation by checking lenient
    boundaries (start/end of token, not strict `\b` regex).
    """
    q = " " + query.lower() + " "
    found: list[str] = []
    consumed = q
    # Longest first so "cdk4/6 inhibitor" wins over "cdk4/6", etc.
    for alias in sorted(_DRUG_CLASS_QUERY_ALIASES, key=len, reverse=True):
        # Build a lenient boundary: alias must be preceded and followed by a
        # non-alphanumeric character (or string edge). This handles "ADC +"
        # and "ADC," correctly without false-matching inside "advance".
        pattern = re.compile(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
            re.IGNORECASE,
        )
        if pattern.search(consumed):
            label = _DRUG_CLASS_QUERY_ALIASES[alias]
            if label not in found:
                found.append(label)
            consumed = pattern.sub(" ", consumed)
    return found


def _expand_query_with_drug_classes(query: str, classes: list[str]) -> str:
    """Append representative drug names from each class to the retrieval query.

    The original query is preserved verbatim; expansions are appended so the
    embedder sees the user's wording first and concrete drug names as
    additional anchors. Skip drugs whose names already appear in the query
    to keep the expansion compact.
    """
    if not classes:
        return query
    qlow = query.lower()
    extras: list[str] = []
    for cls in classes:
        for drug in _DRUG_CLASS_EXPANSION.get(cls, []):
            if drug.lower() not in qlow and drug not in extras:
                extras.append(drug)
    if not extras:
        return query
    return query + " " + " ".join(extras)


def _doc_matches_drug_classes(doc: dict, target_classes: list[str], require_all: bool) -> bool:
    """True if doc's classified drug classes overlap with the target list.

    `require_all=True` enforces an intersection (e.g. "ADC + checkpoint" returns
    only trials with BOTH classes); `False` accepts any-match (used as a
    fallback when intersection is empty).
    """
    if not target_classes:
        return True
    doc_classes = set(classify_trial(doc))
    target = set(target_classes)
    if require_all:
        return target.issubset(doc_classes)
    return bool(doc_classes & target)


# ---- Treatment-setting filter (adjuvant / neoadjuvant / metastatic) ----
#
# Breast cancer trials fall into distinct clinical settings:
#   - adjuvant / neoadjuvant: early-stage disease, treatment around surgery
#   - metastatic (a.k.a. "advanced", stage IV, mBC): incurable disseminated disease
# "Advanced" is a clinical synonym for metastatic, so a query for adjuvant trials
# should not surface trials whose population is advanced/metastatic.

_METASTATIC_RE = re.compile(r"\b(metastatic|advanced|stage\s*iv|stage\s*4|mbc)\b", re.I)
_EARLY_RE = re.compile(r"\b(early|early-stage|non-metastatic|ebc|stage\s*i\b|stage\s*ii\b|stage\s*iii\b)", re.I)
_ADJUVANT_RE = re.compile(r"(?<!neo)\badjuvant\b", re.I)
_NEOADJUVANT_RE = re.compile(r"\bneoadjuvant\b", re.I)


def _extract_setting_intent(query: str) -> str | None:
    """Detect treatment-setting intent in the query.

    Returns one of {"neoadjuvant", "adjuvant", "metastatic"} or None.
    Neoadjuvant is checked before adjuvant because "adjuvant" is a substring match.
    """
    if _NEOADJUVANT_RE.search(query):
        return "neoadjuvant"
    if _ADJUVANT_RE.search(query) or re.search(r"\b(early|early-stage|ebc|non-metastatic)\b", query, re.I):
        return "adjuvant"
    if _METASTATIC_RE.search(query):
        return "metastatic"
    return None


def _doc_setting_text(doc: dict) -> str:
    """Concatenate the high-signal fields (title/conditions/keywords) for a doc."""
    md = doc.get("metadata", {})
    return " ".join([
        md.get("title", ""),
        md.get("officialTitle", ""),
        md.get("conditions", ""),
        md.get("keywords", ""),
    ])


def _doc_matches_setting(doc: dict, intent: str) -> bool:
    """True if doc's title/conditions are consistent with the requested setting."""
    text = _doc_setting_text(doc)
    has_met = bool(_METASTATIC_RE.search(text))
    has_early = bool(_EARLY_RE.search(text))
    has_adj = bool(_ADJUVANT_RE.search(text))
    has_neo = bool(_NEOADJUVANT_RE.search(text))

    if intent == "metastatic":
        return has_met
    if intent == "neoadjuvant":
        return has_neo
    if intent == "adjuvant":
        # Keep trials that explicitly signal adjuvant/neoadjuvant/early-stage.
        # Exclude trials whose primary population is metastatic/advanced
        # unless they also carry an explicit adjuvant or early-stage signal.
        if has_adj or has_neo or has_early:
            return True
        return not has_met
    return True


class ClinicalTrialRAG:
    """End-to-end RAG pipeline for clinical trial Q&A."""

    def __init__(
        self,
        csv_path: str,
        llm: BaseLLMProvider,
        retriever_type: str = "voyage",
        voyage_api_key: str = "",
    ):
        logger.info("Loading clinical trial data from %s", csv_path)
        self.df = load_clinical_trials(csv_path)
        self.documents = build_documents(self.df)

        if retriever_type == "voyage" and voyage_api_key:
            logger.info("Using Voyage AI dense retriever")
            self.retriever = VoyageRetriever(self.documents, api_key=voyage_api_key)
        else:
            if retriever_type == "voyage" and not voyage_api_key:
                logger.warning("Voyage API key not set — falling back to TF-IDF retriever")
            logger.info("Using TF-IDF sparse retriever")
            self.retriever = TFIDFRetriever(self.documents)

        self.llm = llm
        logger.info(
            "RAG pipeline ready (%d trials indexed, llm=%s/%s)",
            len(self.documents), llm.name, llm.model,
        )

    @property
    def model(self) -> str:
        """Backward-compat shim: some call sites and the /health endpoint
        still read pipeline.model. Delegates to the active provider."""
        return self.llm.model

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_scores: dict[str, float] | None = None,
    ) -> list[dict]:
        """Retrieve relevant trials without generating an answer.

        ``source_scores`` (when provided) nudges retrieval order based on past
        user feedback per NCT id — see ``core.feedback_reranker``."""
        return self._retrieve_with_country_filter(query, top_k, source_scores)

    def _retrieve_with_country_filter(
        self,
        query: str,
        top_k: int,
        source_scores: dict[str, float] | None = None,
    ) -> list[dict]:
        """Retrieve top_k docs, hard-filtering by the structured attributes named in the query."""
        countries = _extract_countries(query)
        states = _extract_us_states(query)
        setting = _extract_setting_intent(query)
        study_type = _extract_study_type(query)
        statuses = _extract_statuses(query)
        sponsors = _extract_sponsors(query)
        drug_classes = _extract_drug_classes(query)

        if not any([countries, states, setting, study_type, statuses, sponsors, drug_classes]):
            # Over-fetch a modest pool so the feedback reranker has something
            # to work with; otherwise it can only re-order an already-final
            # top_k. When there's no feedback, this is a single extra slice.
            pool_size = max(top_k * 3, 20) if source_scores else top_k
            results = self.retriever.retrieve(query, top_k=pool_size)
            return adjust_scores(results, source_scores)[:top_k]

        # Drug-class filters need both expansion (so dense retrieval finds the
        # specific drugs) and an aggressive pool — required-intersection
        # filtering on multiple classes is the strictest filter we apply.
        if study_type or statuses or sponsors or drug_classes:
            pool_size = max(top_k * 20, 200)
        else:
            pool_size = max(top_k * 10, 50)

        # Expand the retrieval query with canonical drug names from any
        # detected classes so the embedder sees concrete anchors.
        retrieval_query = _expand_query_with_drug_classes(query, drug_classes)
        pool = self.retriever.retrieve(retrieval_query, top_k=pool_size)
        filtered = pool
        if countries:
            filtered = [d for d in filtered if _doc_matches_countries(d, countries)]
        if states:
            filtered = [d for d in filtered if _doc_matches_states(d, states)]
        if setting:
            filtered = [d for d in filtered if _doc_matches_setting(d, setting)]
        if study_type:
            filtered = [d for d in filtered if _doc_matches_study_type(d, study_type)]
        if statuses:
            filtered = [d for d in filtered if _doc_matches_statuses(d, statuses)]
        if sponsors:
            filtered = [d for d in filtered if _doc_matches_sponsors(d, sponsors)]
        if drug_classes:
            # Strict pass: trials must match ALL requested classes (intersection).
            strict = [d for d in filtered if _doc_matches_drug_classes(d, drug_classes, require_all=True)]
            if strict:
                filtered = strict
            else:
                # Fallback: any class match. Better than dropping the filter.
                lenient = [d for d in filtered if _doc_matches_drug_classes(d, drug_classes, require_all=False)]
                filtered = lenient or filtered

        filter_summary = (
            f"countries={countries} states={states} setting={setting} "
            f"studyType={study_type} statuses={statuses} sponsors={sponsors} "
            f"drugClasses={drug_classes}"
        )
        if not filtered:
            logger.info("Filter (%s) matched 0 trials — falling back to unfiltered results", filter_summary)
            return adjust_scores(pool, source_scores)[:top_k]
        logger.info("Filter (%s) kept %d / %d retrieved trials", filter_summary, len(filtered), len(pool))
        return adjust_scores(filtered, source_scores)[:top_k]

    async def ask(
        self,
        query: str,
        top_k: int = 5,
        source_scores: dict[str, float] | None = None,
    ) -> tuple[str, list[dict]]:
        """Retrieve trials and generate a full answer. Returns (answer, sources)."""
        retrieved = self._retrieve_with_country_filter(query, top_k, source_scores)
        if not retrieved:
            return "No relevant clinical trials found for your query.", []
        answer = await generate_answer(self.llm, query, retrieved)
        return answer, retrieved

    async def ask_stream(
        self,
        query: str,
        top_k: int = 5,
        source_scores: dict[str, float] | None = None,
    ):
        """Retrieve trials and stream answer tokens. Returns (stream_generator, sources)."""
        retrieved = self._retrieve_with_country_filter(query, top_k, source_scores)
        if not retrieved:
            return None, []
        stream = generate_answer_stream(self.llm, query, retrieved)
        return stream, retrieved
