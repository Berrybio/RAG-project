"""
Pull breast cancer clinical trial data from ClinicalTrials.gov API (v2).

Usage:
    python -m scripts.pull_clinical_trials                          # all statuses, all phases
    python -m scripts.pull_clinical_trials --status RECRUITING      # only recruiting
    python -m scripts.pull_clinical_trials --phase phase:2          # only phase 2
    python -m scripts.pull_clinical_trials --dry-run                # fetch first page only

Output is saved to backend/data/ as a timestamped CSV.
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, fields, asdict
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CT_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_PAGE_SIZE = 100          # max allowed by the API
REQUEST_DELAY = 0.3              # seconds between pages (be polite)
MAX_RETRIES = 3
RETRY_DELAY = 5                  # seconds between retries


# ---------------------------------------------------------------------------
# Data model — all fields available from ClinicalTrials.gov
# ---------------------------------------------------------------------------
@dataclass
class CTData:
    # --- Identification ---
    nctId: str = ""
    orgID: str = ""
    acronym: str = ""
    bTitle: str = ""
    oTitle: str = ""

    # --- Sponsor & Organization ---
    sponsorName: str = ""
    sponsorClass: str = ""          # INDUSTRY, NIH, FED, OTHER
    collaborators: str = ""
    responsiblePartyType: str = ""  # SPONSOR, PRINCIPAL_INVESTIGATOR, etc.

    # --- Status & Dates ---
    oStatus: str = ""
    statusBTS: str = ""             # statusVerifiedDate
    startDate: str = ""
    startDateType: str = ""         # ACTUAL or ESTIMATED
    primaryCompletionDate: str = ""
    primaryCompletionDateType: str = ""
    completionETA: str = ""
    completionETAType: str = ""
    studyFST: str = ""             # studyFirstSubmitDate
    lastUpdateDate: str = ""

    # --- Description ---
    summary: str = ""
    description: str = ""

    # --- Conditions & Keywords ---
    conditions: str = ""
    keywords: str = ""
    meshTermsCondition: str = ""    # standardized medical vocabulary
    meshTermsIntervention: str = ""

    # --- Study Design ---
    studyType: str = ""             # INTERVENTIONAL, OBSERVATIONAL, etc.
    phases: str = ""
    allocation: str = ""            # RANDOMIZED, NON_RANDOMIZED, N/A
    interventionModel: str = ""     # PARALLEL, CROSSOVER, SEQUENTIAL, etc.
    primaryPurpose: str = ""        # TREATMENT, PREVENTION, DIAGNOSTIC, etc.
    masking: str = ""               # NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE
    enrollmentCont: int = 0
    enrollmentType: str = ""        # ACTUAL or ESTIMATED

    # --- Arms & Interventions ---
    armGroups: str = ""             # arm labels, types, descriptions
    interventionName: str = ""
    interventionType: str = ""      # DRUG, DEVICE, PROCEDURE, etc.
    interventionDescription: str = ""
    interventionOtherNames: str = ""  # drug aliases (e.g., IBRANCE for Palbociclib)

    # --- Eligibility ---
    includeCriteria: str = ""
    excludeCriteria: str = ""
    sex: str = "ALL"
    minimumAge: str = ""
    maximumAge: str = ""
    healthyVolunteers: str = ""
    stdAges: str = ""

    # --- Outcomes ---
    primaryOutcomes: str = ""
    secondaryOutcomes: str = ""

    # --- Contacts & Locations ---
    piName: str = ""                # principal investigator
    piAffiliation: str = ""
    contactInfo: str = ""
    locationInfo: str = ""
    locationCount: int = 0
    locationCountries: str = ""

    # --- Oversight ---
    isFdaRegulatedDrug: str = ""
    isFdaRegulatedDevice: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split_criteria(text: str) -> tuple[str, str]:
    """Split eligibility criteria into inclusion and exclusion sections."""
    inclusion_marker = "Inclusion Criteria"
    exclusion_marker = "Exclusion Criteria"

    inclusion_start = text.find(inclusion_marker)
    exclusion_start = text.find(exclusion_marker)

    if inclusion_start == -1:
        inclusion_start = 0

    inclusion_criteria = (
        text[inclusion_start:exclusion_start].strip()
        if exclusion_start != -1
        else text[inclusion_start:].strip()
    )
    exclusion_criteria = (
        text[exclusion_start:].strip() if exclusion_start != -1 else ""
    )
    return inclusion_criteria, exclusion_criteria


def parse_study(study: dict) -> CTData:
    """Parse a single study JSON into a CTData object."""
    protocol = study.get("protocolSection", {})
    derived = study.get("derivedSection", {})

    ident = protocol.get("identificationModule", {})
    status_mod = protocol.get("statusModule", {})
    desc = protocol.get("descriptionModule", {})
    cond = protocol.get("conditionsModule", {})
    design = protocol.get("designModule", {})
    elig = protocol.get("eligibilityModule", {})
    contacts = protocol.get("contactsLocationsModule", {})
    arms_mod = protocol.get("armsInterventionsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
    oversight = protocol.get("oversightModule", {})

    # --- Sponsor ---
    lead_sponsor = sponsor_mod.get("leadSponsor", {})
    collaborators_list = sponsor_mod.get("collaborators", [])
    collaborators_str = ", ".join(
        f"{c.get('name', '')} ({c.get('class', '')})"
        for c in collaborators_list
    )
    responsible_party = sponsor_mod.get("responsibleParty", {})

    # --- Interventions ---
    interventions = arms_mod.get("interventions", [])
    intervention_names = ", ".join(
        i.get("name", "") for i in interventions if i.get("name")
    )
    intervention_types = ", ".join(
        i.get("type", "") for i in interventions if i.get("type")
    )
    intervention_descs = ", ".join(
        i.get("description", "") for i in interventions if i.get("description")
    )
    # Collect all drug aliases / other names
    other_names = []
    for i in interventions:
        other_names.extend(i.get("otherNames", []))
    intervention_other_names = ", ".join(other_names)

    # --- Arm Groups ---
    arm_groups = arms_mod.get("armGroups", [])
    arm_str = " | ".join(
        f"{a.get('label', '')}: {a.get('type', '')} - {a.get('description', '')}"
        for a in arm_groups
    )

    # --- Eligibility criteria ---
    raw_criteria = elig.get("eligibilityCriteria", "")
    include, exclude = split_criteria(raw_criteria)

    # --- Outcomes ---
    primary_outcomes = outcomes.get("primaryOutcomes", [])
    primary_str = " | ".join(
        f"{o.get('measure', '')}: {o.get('description', '')} [{o.get('timeFrame', '')}]"
        for o in primary_outcomes
    )
    secondary_outcomes = outcomes.get("secondaryOutcomes", [])
    secondary_str = " | ".join(
        f"{o.get('measure', '')}: {o.get('description', '')} [{o.get('timeFrame', '')}]"
        for o in secondary_outcomes
    )

    # --- PI / Overall Officials ---
    officials = contacts.get("overallOfficials", [])
    pi_name = ""
    pi_affiliation = ""
    for off in officials:
        if off.get("role") == "PRINCIPAL_INVESTIGATOR":
            pi_name = off.get("name", "")
            pi_affiliation = off.get("affiliation", "")
            break
    # If no PI found, take the first official
    if not pi_name and officials:
        pi_name = officials[0].get("name", "")
        pi_affiliation = officials[0].get("affiliation", "")

    # --- Central Contacts ---
    central_contacts = contacts.get("centralContacts", [])
    contact_str = ", ".join(
        f"{c.get('name', 'N/A')} ({c.get('role', 'N/A')}) - {c.get('email', 'N/A')}"
        for c in central_contacts
    )

    # --- Locations ---
    locations = contacts.get("locations", [])
    location_str = ", ".join(
        f"{loc.get('facility', 'N/A')} ({loc.get('status', 'N/A')}) "
        f"- {loc.get('city', 'N/A')}, {loc.get('state', 'N/A')}, {loc.get('country', 'N/A')}"
        for loc in locations
    )
    # Unique countries
    countries = sorted(set(
        loc.get("country", "") for loc in locations if loc.get("country")
    ))

    # --- Study Design ---
    design_info = design.get("designInfo", {})
    masking_info = design_info.get("maskingInfo", {})
    enrollment_info = design.get("enrollmentInfo", {})

    # --- MeSH terms from derived section ---
    cond_browse = derived.get("conditionBrowseModule", {})
    intv_browse = derived.get("interventionBrowseModule", {})
    mesh_conditions = ", ".join(
        m.get("term", "") for m in cond_browse.get("meshes", [])
    )
    mesh_interventions = ", ".join(
        m.get("term", "") for m in intv_browse.get("meshes", [])
    )

    return CTData(
        # Identification
        nctId=ident.get("nctId", ""),
        orgID=ident.get("orgStudyIdInfo", {}).get("id", ""),
        acronym=ident.get("acronym", ""),
        bTitle=ident.get("briefTitle", ""),
        oTitle=ident.get("officialTitle", ""),
        # Sponsor
        sponsorName=lead_sponsor.get("name", ""),
        sponsorClass=lead_sponsor.get("class", ""),
        collaborators=collaborators_str,
        responsiblePartyType=responsible_party.get("type", ""),
        # Status & Dates
        oStatus=status_mod.get("overallStatus", ""),
        statusBTS=status_mod.get("statusVerifiedDate", ""),
        startDate=status_mod.get("startDateStruct", {}).get("date", ""),
        startDateType=status_mod.get("startDateStruct", {}).get("type", ""),
        primaryCompletionDate=status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
        primaryCompletionDateType=status_mod.get("primaryCompletionDateStruct", {}).get("type", ""),
        completionETA=status_mod.get("completionDateStruct", {}).get("date", ""),
        completionETAType=status_mod.get("completionDateStruct", {}).get("type", ""),
        studyFST=status_mod.get("studyFirstSubmitDate", ""),
        lastUpdateDate=status_mod.get("lastUpdatePostDateStruct", {}).get("date", ""),
        # Description
        summary=desc.get("briefSummary", ""),
        description=desc.get("detailedDescription", ""),
        # Conditions & Keywords
        conditions=", ".join(cond.get("conditions", [])),
        keywords=", ".join(cond.get("keywords", [])),
        meshTermsCondition=mesh_conditions,
        meshTermsIntervention=mesh_interventions,
        # Study Design
        studyType=design.get("studyType", ""),
        phases=", ".join(design.get("phases", [])),
        allocation=design_info.get("allocation", ""),
        interventionModel=design_info.get("interventionModel", ""),
        primaryPurpose=design_info.get("primaryPurpose", ""),
        masking=masking_info.get("masking", ""),
        enrollmentCont=enrollment_info.get("count", 0),
        enrollmentType=enrollment_info.get("type", ""),
        # Arms & Interventions
        armGroups=arm_str,
        interventionName=intervention_names,
        interventionType=intervention_types,
        interventionDescription=intervention_descs,
        interventionOtherNames=intervention_other_names,
        # Eligibility
        includeCriteria=include,
        excludeCriteria=exclude,
        sex=elig.get("sex", "ALL"),
        minimumAge=elig.get("minimumAge", ""),
        maximumAge=elig.get("maximumAge", ""),
        healthyVolunteers=str(elig.get("healthyVolunteers", "")),
        stdAges=", ".join(elig.get("stdAges", [])),
        # Outcomes
        primaryOutcomes=primary_str,
        secondaryOutcomes=secondary_str,
        # Contacts & Locations
        piName=pi_name,
        piAffiliation=pi_affiliation,
        contactInfo=contact_str,
        locationInfo=location_str,
        locationCount=len(locations),
        locationCountries=", ".join(countries),
        # Oversight
        isFdaRegulatedDrug=str(oversight.get("isFdaRegulatedDrug", "")),
        isFdaRegulatedDevice=str(oversight.get("isFdaRegulatedDevice", "")),
    )


# ---------------------------------------------------------------------------
# API fetching with pagination and retries
# ---------------------------------------------------------------------------
def fetch_page(params: dict, retry: int = 0) -> dict | None:
    """Fetch a single page from the ClinicalTrials.gov API."""
    try:
        response = requests.get(CT_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if retry < MAX_RETRIES:
            print(f"  ⚠ Request failed ({e}), retrying in {RETRY_DELAY}s... ({retry + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            return fetch_page(params, retry + 1)
        print(f"  ✗ Failed after {MAX_RETRIES} retries: {e}")
        return None


def pull_all_studies(
    condition: str = "Breast Cancer",
    status: str | None = None,
    phase: str | None = None,
    dry_run: bool = False,
) -> list[CTData]:
    """Pull all matching studies with pagination."""

    params = {
        "query.cond": condition,
        "pageSize": DEFAULT_PAGE_SIZE,
    }
    if status:
        params["filter.overallStatus"] = status
    if phase:
        params["aggFilters"] = phase

    all_trials = []
    page_num = 1

    while True:
        print(f"📄 Fetching page {page_num} (total so far: {len(all_trials)})...")
        data = fetch_page(params)

        if not data or "studies" not in data:
            print("  ✗ No data returned, stopping.")
            break

        studies = data["studies"]
        for study in studies:
            trial = parse_study(study)
            all_trials.append(trial)

        print(f"  ✓ Got {len(studies)} studies (total: {len(all_trials)})")

        if dry_run:
            print("  🛑 Dry run — stopping after first page.")
            break

        next_page = data.get("nextPageToken")
        if not next_page:
            print("✅ No more pages — done!")
            break

        params["pageToken"] = next_page
        page_num += 1
        time.sleep(REQUEST_DELAY)

    return all_trials


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def save_to_csv(trials: list[CTData], output_path: str):
    """Save trial data to CSV."""
    fieldnames = [f.name for f in fields(CTData)]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow(asdict(trial))

    print(f"💾 Saved {len(trials)} trials to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pull breast cancer clinical trial data from ClinicalTrials.gov"
    )
    parser.add_argument(
        "--condition", default="Breast Cancer",
        help="Condition to search for (default: 'Breast Cancer')"
    )
    parser.add_argument(
        "--status", default=None,
        help="Filter by status, e.g. RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING. "
             "Omit for all statuses."
    )
    parser.add_argument(
        "--phase", default=None,
        help="Filter by phase, e.g. 'phase:2', 'phase:3'. Omit for all phases."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Defaults to backend/data/breast_cancer_trials_YYYY-MM-DD.csv"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch only the first page (for testing)"
    )
    args = parser.parse_args()

    # Build output filename
    if args.output:
        output_path = args.output
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = os.path.join(data_dir, f"breast_cancer_trials_{date_str}.csv")

    print("=" * 60)
    print("ClinicalTrials.gov Data Puller")
    print("=" * 60)
    print(f"  Condition : {args.condition}")
    print(f"  Status    : {args.status or 'ALL'}")
    print(f"  Phase     : {args.phase or 'ALL'}")
    print(f"  Output    : {output_path}")
    print(f"  Dry run   : {args.dry_run}")
    print("=" * 60)

    trials = pull_all_studies(
        condition=args.condition,
        status=args.status,
        phase=args.phase,
        dry_run=args.dry_run,
    )

    if trials:
        save_to_csv(trials, output_path)
        print(f"\n📊 Summary:")
        print(f"   Total trials: {len(trials)}")
        print(f"   Total columns: {len(fields(CTData))}")

        # Status breakdown
        statuses = {}
        for t in trials:
            statuses[t.oStatus] = statuses.get(t.oStatus, 0) + 1
        print(f"   By status:")
        for s, count in sorted(statuses.items(), key=lambda x: -x[1]):
            print(f"     {s}: {count}")

        # Phase breakdown
        phases = {}
        for t in trials:
            phases[t.phases or "N/A"] = phases.get(t.phases or "N/A", 0) + 1
        print(f"   By phase:")
        for p, count in sorted(phases.items(), key=lambda x: -x[1]):
            print(f"     {p}: {count}")

        # Study type breakdown
        study_types = {}
        for t in trials:
            study_types[t.studyType or "N/A"] = study_types.get(t.studyType or "N/A", 0) + 1
        print(f"   By study type:")
        for st, count in sorted(study_types.items(), key=lambda x: -x[1]):
            print(f"     {st}: {count}")

        # Sponsor class breakdown
        sponsor_classes = {}
        for t in trials:
            sponsor_classes[t.sponsorClass or "N/A"] = sponsor_classes.get(t.sponsorClass or "N/A", 0) + 1
        print(f"   By sponsor class:")
        for sc, count in sorted(sponsor_classes.items(), key=lambda x: -x[1]):
            print(f"     {sc}: {count}")
    else:
        print("⚠ No trials found.")


if __name__ == "__main__":
    main()
