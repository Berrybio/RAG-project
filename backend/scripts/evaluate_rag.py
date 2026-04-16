"""
Evaluation suite for the Clinical Trial RAG pipeline.

Evaluates both retrieval quality and generation quality using a curated
set of test queries with expected trial matches.

Usage:
    python -m scripts.evaluate_rag                              # full evaluation
    python -m scripts.evaluate_rag --retrieval-only             # skip LLM generation
    python -m scripts.evaluate_rag --csv data/custom.csv        # custom dataset
    python -m scripts.evaluate_rag --top-k 10                   # evaluate at k=10

Metrics:
    Retrieval:
        - Hit Rate (Recall@k):  fraction of queries where >= 1 expected trial is in top-k
        - MRR (Mean Reciprocal Rank): average of 1/rank of the first correct result
        - Precision@k: average fraction of top-k results that are relevant
        - MAP@k (Mean Average Precision): average precision across recall points

    Generation (requires ANTHROPIC_API_KEY):
        - Faithfulness: does the answer only use information from retrieved context?
        - Relevance: does the answer address the user's question?
        - Completeness: does the answer reference the expected trials?
        - Overall quality: average of above scores

Output is saved to backend/data/eval_results_YYYY-MM-DD.json
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import anthropic

from app.core.data import load_clinical_trials, build_documents
from app.core.retriever import TFIDFRetriever, VoyageRetriever
from app.core.generation import generate_answer, format_context

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test queries with expected results
# Each query has:
#   - query: the user's natural language question
#   - expected_nct_ids: NCT IDs that should appear in top-k (at least one)
#   - category: what type of query this is (for breakdown reporting)
#   - description: why this query is a good test
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    # --- Drug-specific queries ---
    {
        "query": "CDK4/6 inhibitor trials for hormone receptor positive breast cancer",
        "expected_nct_ids": ["NCT04567420"],
        "category": "drug_specific",
        "description": "Should find CDK4/6 inhibitor trials like palbociclib, ribociclib",
    },
    {
        "query": "pembrolizumab immunotherapy trials for triple negative breast cancer",
        "expected_nct_ids": [],  # will match any TNBC + pembrolizumab trial
        "category": "drug_specific",
        "expected_keywords": ["pembrolizumab", "triple negative", "TNBC"],
        "description": "Should find checkpoint inhibitor trials for TNBC",
    },
    {
        "query": "trastuzumab deruxtecan T-DXd for HER2 positive breast cancer",
        "expected_nct_ids": [],
        "category": "drug_specific",
        "expected_keywords": ["trastuzumab", "HER2", "T-DXd", "Enhertu"],
        "description": "Tests drug alias matching (T-DXd = trastuzumab deruxtecan = Enhertu)",
    },
    {
        "query": "PARP inhibitor olaparib trials for BRCA mutated breast cancer",
        "expected_nct_ids": [],
        "category": "drug_specific",
        "expected_keywords": ["PARP", "olaparib", "BRCA"],
        "description": "Tests targeted therapy for biomarker-defined population",
    },
    # --- Condition-specific queries ---
    {
        "query": "clinical trials for inflammatory breast cancer",
        "expected_nct_ids": [],
        "category": "condition_specific",
        "expected_keywords": ["inflammatory", "breast cancer"],
        "description": "Rare subtype query",
    },
    {
        "query": "trials for breast cancer that has spread to the brain",
        "expected_nct_ids": [],
        "category": "condition_specific",
        "expected_keywords": ["brain metast", "CNS"],
        "description": "Tests semantic understanding (spread to brain = brain metastases)",
    },
    {
        "query": "neoadjuvant treatment trials before breast cancer surgery",
        "expected_nct_ids": [],
        "category": "condition_specific",
        "expected_keywords": ["neoadjuvant"],
        "description": "Tests treatment setting terminology",
    },
    {
        "query": "male breast cancer clinical trials",
        "expected_nct_ids": [],
        "category": "condition_specific",
        "expected_keywords": ["male", "breast cancer"],
        "description": "Tests rare population query",
    },
    # --- Eligibility queries ---
    {
        "query": "breast cancer trials that accept patients over 70 years old",
        "expected_nct_ids": [],
        "category": "eligibility",
        "expected_keywords": ["age", "elderly", "older"],
        "description": "Tests eligibility-focused retrieval",
    },
    {
        "query": "breast cancer trials for pregnant women",
        "expected_nct_ids": [],
        "category": "eligibility",
        "expected_keywords": ["pregnant", "pregnancy"],
        "description": "Tests specific eligibility constraint",
    },
    # --- Design queries ---
    {
        "query": "randomized phase 3 breast cancer trials",
        "expected_nct_ids": [],
        "category": "study_design",
        "expected_keywords": ["PHASE3", "RANDOMIZED"],
        "description": "Tests study design filtering",
    },
    {
        "query": "double blind placebo controlled breast cancer studies",
        "expected_nct_ids": [],
        "category": "study_design",
        "expected_keywords": ["DOUBLE", "placebo"],
        "description": "Tests masking/blinding terminology",
    },
    # --- Location queries ---
    {
        "query": "breast cancer trials in Texas recruiting now",
        "expected_nct_ids": [],
        "category": "location",
        "expected_keywords": ["Texas", "RECRUITING"],
        "description": "Tests location + status filtering",
    },
    {
        "query": "breast cancer clinical trials available in Europe",
        "expected_nct_ids": [],
        "category": "location",
        "expected_keywords": ["Germany", "France", "United Kingdom", "Italy", "Spain"],
        "description": "Tests geographic region query",
    },
    # --- Sponsor / PI queries ---
    {
        "query": "breast cancer trials sponsored by Pfizer",
        "expected_nct_ids": [],
        "category": "sponsor",
        "expected_keywords": ["Pfizer"],
        "description": "Tests sponsor-based retrieval",
    },
    {
        "query": "industry-sponsored breast cancer trials",
        "expected_nct_ids": [],
        "category": "sponsor",
        "expected_keywords": ["INDUSTRY"],
        "description": "Tests sponsor class retrieval",
    },
    # --- Outcome queries ---
    {
        "query": "breast cancer trials measuring overall survival as primary endpoint",
        "expected_nct_ids": [],
        "category": "outcomes",
        "expected_keywords": ["overall survival", "OS"],
        "description": "Tests outcome-based retrieval",
    },
    {
        "query": "breast cancer trials with pathological complete response endpoint",
        "expected_nct_ids": [],
        "category": "outcomes",
        "expected_keywords": ["pathological complete response", "pCR"],
        "description": "Tests specific endpoint terminology",
    },
    # --- Natural language / semantic queries ---
    {
        "query": "what treatments are available for early stage breast cancer",
        "expected_nct_ids": [],
        "category": "semantic",
        "expected_keywords": ["early", "stage", "adjuvant"],
        "description": "Tests natural language understanding",
    },
    {
        "query": "new experimental drugs being tested for metastatic breast cancer",
        "expected_nct_ids": [],
        "category": "semantic",
        "expected_keywords": ["metastatic", "experimental", "DRUG"],
        "description": "Tests colloquial query understanding",
    },
]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    query: str
    category: str
    top_k_ids: list[str]
    top_k_scores: list[float]
    expected_nct_ids: list[str]
    expected_keywords: list[str]
    hit: bool                  # at least one expected trial or keyword found
    reciprocal_rank: float     # 1/rank of first correct result (0 if none)
    precision: float           # fraction of top-k that match expected
    keyword_hits: dict         # which keywords were found in which results


def evaluate_retrieval_single(
    retriever: TFIDFRetriever,
    query_item: dict,
    top_k: int,
) -> RetrievalResult:
    """Evaluate a single retrieval query."""
    results = retriever.retrieve(query_item["query"], top_k=top_k)
    top_k_ids = [r["doc_id"] for r in results]
    top_k_scores = [r["score"] for r in results]

    expected_ids = query_item.get("expected_nct_ids", [])
    expected_keywords = query_item.get("expected_keywords", [])

    # Check NCT ID hits
    id_hit = False
    first_rank = 0
    id_matches = 0
    for rank, doc_id in enumerate(top_k_ids, 1):
        if doc_id in expected_ids:
            if not id_hit:
                first_rank = rank
                id_hit = True
            id_matches += 1

    # Check keyword hits in retrieved text
    keyword_hits = {}
    keyword_hit = False
    for kw in expected_keywords:
        kw_lower = kw.lower()
        for rank, result in enumerate(results, 1):
            text = result["text"].lower()
            meta_str = json.dumps(result["metadata"]).lower()
            if kw_lower in text or kw_lower in meta_str:
                keyword_hits[kw] = rank
                keyword_hit = True
                break

    # Determine overall hit
    if expected_ids:
        hit = id_hit
    else:
        # If no specific IDs, check if at least half the keywords matched
        hit = len(keyword_hits) >= max(1, len(expected_keywords) // 2)

    # Reciprocal rank
    if first_rank > 0:
        rr = 1.0 / first_rank
    elif keyword_hits:
        best_rank = min(keyword_hits.values())
        rr = 1.0 / best_rank
    else:
        rr = 0.0

    # Precision: for keyword-based eval, count results containing any keyword
    if expected_keywords:
        relevant_count = 0
        for result in results:
            text = result["text"].lower()
            meta_str = json.dumps(result["metadata"]).lower()
            if any(kw.lower() in text or kw.lower() in meta_str for kw in expected_keywords):
                relevant_count += 1
        precision = relevant_count / len(results) if results else 0.0
    elif expected_ids:
        precision = id_matches / len(results) if results else 0.0
    else:
        precision = 0.0

    return RetrievalResult(
        query=query_item["query"],
        category=query_item["category"],
        top_k_ids=top_k_ids,
        top_k_scores=top_k_scores,
        expected_nct_ids=expected_ids,
        expected_keywords=expected_keywords,
        hit=hit,
        reciprocal_rank=rr,
        precision=precision,
        keyword_hits=keyword_hits,
    )


# ---------------------------------------------------------------------------
# Generation quality evaluation (LLM-as-judge)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of clinical trial question-answering systems.
You will be given a user query, retrieved clinical trial context, and a
generated answer. Score the answer on three dimensions.

IMPORTANT: The context contains multiple trials enclosed in <trial> XML tags.
Each <trial> block is a separate trial that WAS retrieved from the database.
Count the actual <trial> tags to determine how many trials were provided.
The answer may legitimately reference ALL trials present in the context.

Return your evaluation as JSON with exactly these keys:
{
  "faithfulness": <1-5>,
  "faithfulness_reason": "brief explanation",
  "relevance": <1-5>,
  "relevance_reason": "brief explanation",
  "completeness": <1-5>,
  "completeness_reason": "brief explanation"
}

Scoring rubric:

Faithfulness (is every claim in the answer supported by the provided context?):
  5 = Every claim maps directly to a field in the <trial> data
  4 = Almost all claims supported, minor reasonable inferences from trial fields
  3 = Mostly supported but includes some details not found in the context
  2 = Multiple claims that cannot be traced to the provided trial data
  1 = Fabricates trial details, drugs, or results not in any <trial> block

Relevance (does the answer address the question?):
  5 = Directly and thoroughly answers the question
  4 = Answers the question with minor gaps
  3 = Partially answers the question
  2 = Tangentially related
  1 = Does not answer the question

Completeness (does it cover the key trials and information?):
  5 = Covers all provided trials with key details (phase, status, intervention, outcomes)
  4 = Good coverage with minor omissions
  3 = Covers main points but misses important details from the trial data
  2 = Significant gaps — skips trials or omits critical fields
  1 = Very incomplete

Return ONLY the JSON object, no other text."""


@dataclass
class GenerationResult:
    query: str
    category: str
    answer: str
    faithfulness: int
    faithfulness_reason: str
    relevance: int
    relevance_reason: str
    completeness: int
    completeness_reason: str
    latency_seconds: float


async def evaluate_generation_single(
    client: anthropic.AsyncAnthropic,
    query: str,
    retrieved_docs: list[dict],
    answer: str,
    category: str,
    judge_model: str = "claude-sonnet-4-20250514",
) -> GenerationResult:
    """Use LLM-as-judge to evaluate a single generated answer."""
    context = format_context(retrieved_docs)

    n_trials = len(retrieved_docs)
    nct_ids = [doc["metadata"].get("nctId", "?") for doc in retrieved_docs]
    judge_msg = (
        f"User Query: {query}\n\n"
        f"Number of trials in context: {n_trials}\n"
        f"NCT IDs in context: {', '.join(nct_ids)}\n\n"
        f"=== RETRIEVED CONTEXT ===\n{context}\n=== END ===\n\n"
        f"=== GENERATED ANSWER ===\n{answer}\n=== END ===\n\n"
        f"Evaluate the generated answer against the {n_trials} trials above. "
        f"Return your scores as JSON."
    )

    response = await client.messages.create(
        model=judge_model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": judge_msg}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse judge response: %s", raw[:200])
        scores = {
            "faithfulness": 0, "faithfulness_reason": "parse error",
            "relevance": 0, "relevance_reason": "parse error",
            "completeness": 0, "completeness_reason": "parse error",
        }

    return GenerationResult(
        query=query,
        category=category,
        answer=answer[:500],  # truncate for storage
        faithfulness=scores.get("faithfulness", 0),
        faithfulness_reason=scores.get("faithfulness_reason", ""),
        relevance=scores.get("relevance", 0),
        relevance_reason=scores.get("relevance_reason", ""),
        completeness=scores.get("completeness", 0),
        completeness_reason=scores.get("completeness_reason", ""),
        latency_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------
async def run_evaluation(
    csv_path: str,
    top_k: int = 5,
    retrieval_only: bool = False,
    model: str = "claude-sonnet-4-20250514",
    retriever_type: str = "voyage",
    voyage_api_key: str = "",
) -> dict:
    """Run full evaluation and return results dict."""
    print("=" * 60)
    print("RAG Evaluation Suite")
    print("=" * 60)

    # Load data and build retriever
    print(f"\n📄 Loading data from {csv_path}...")
    df = load_clinical_trials(csv_path)
    documents = build_documents(df)

    if retriever_type == "voyage" and voyage_api_key:
        print(f"   Using Voyage AI dense retriever ({VoyageRetriever.EMBED_MODEL})")
        retriever = VoyageRetriever(documents, api_key=voyage_api_key)
    else:
        if retriever_type == "voyage" and not voyage_api_key:
            print("   ⚠ VOYAGE_API_KEY not set — falling back to TF-IDF")
        print("   Using TF-IDF sparse retriever")
        retriever = TFIDFRetriever(documents)
    print(f"   Indexed {len(documents)} trials")

    # ---- Retrieval evaluation ----
    print(f"\n🔍 Evaluating retrieval (top_k={top_k}, {len(EVAL_QUERIES)} queries)...\n")
    retrieval_results = []
    for i, query_item in enumerate(EVAL_QUERIES, 1):
        result = evaluate_retrieval_single(retriever, query_item, top_k)
        retrieval_results.append(result)
        status = "✓" if result.hit else "✗"
        print(
            f"  {status} [{result.category:20s}] "
            f"RR={result.reciprocal_rank:.3f}  "
            f"P@{top_k}={result.precision:.3f}  "
            f"| {result.query[:60]}"
        )
        if not result.hit:
            print(f"    ↳ Top results: {result.top_k_ids[:3]}")
            if result.expected_keywords:
                missing = [kw for kw in result.expected_keywords if kw not in result.keyword_hits]
                if missing:
                    print(f"    ↳ Missing keywords: {missing}")

    # Aggregate retrieval metrics
    total = len(retrieval_results)
    hit_rate = sum(1 for r in retrieval_results if r.hit) / total
    mrr = sum(r.reciprocal_rank for r in retrieval_results) / total
    avg_precision = sum(r.precision for r in retrieval_results) / total

    # Per-category breakdown
    categories = sorted(set(r.category for r in retrieval_results))
    category_metrics = {}
    for cat in categories:
        cat_results = [r for r in retrieval_results if r.category == cat]
        cat_total = len(cat_results)
        category_metrics[cat] = {
            "count": cat_total,
            "hit_rate": sum(1 for r in cat_results if r.hit) / cat_total,
            "mrr": sum(r.reciprocal_rank for r in cat_results) / cat_total,
            "avg_precision": sum(r.precision for r in cat_results) / cat_total,
        }

    print(f"\n{'─' * 60}")
    print(f"📊 Retrieval Results (top_k={top_k}):")
    print(f"   Hit Rate (Recall@{top_k}): {hit_rate:.1%}")
    print(f"   MRR:                        {mrr:.3f}")
    print(f"   Avg Precision@{top_k}:       {avg_precision:.3f}")
    print(f"\n   By Category:")
    for cat, m in sorted(category_metrics.items()):
        print(
            f"     {cat:20s}  Hit={m['hit_rate']:.0%}  "
            f"MRR={m['mrr']:.3f}  "
            f"P@{top_k}={m['avg_precision']:.3f}  "
            f"(n={m['count']})"
        )

    # ---- Generation evaluation ----
    generation_results = []
    generation_metrics = {}

    if not retrieval_only:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("\n⚠ ANTHROPIC_API_KEY not set — skipping generation evaluation.")
        else:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            # Use a subset for generation eval (it's expensive)
            gen_queries = EVAL_QUERIES[:10]
            print(f"\n🤖 Evaluating generation ({len(gen_queries)} queries, model={model})...\n")

            for i, query_item in enumerate(gen_queries, 1):
                query = query_item["query"]
                category = query_item["category"]

                # Retrieve
                retrieved = retriever.retrieve(query, top_k=top_k)
                if not retrieved:
                    print(f"  ✗ No results for: {query[:50]}")
                    continue

                # Generate
                t0 = time.time()
                try:
                    answer = await generate_answer(client, query, retrieved, model)
                except Exception as e:
                    print(f"  ✗ Generation failed for: {query[:50]} ({e})")
                    continue
                latency = time.time() - t0

                # Judge
                try:
                    gen_result = await evaluate_generation_single(
                        client, query, retrieved, answer, category, model,
                    )
                    gen_result.latency_seconds = latency
                    generation_results.append(gen_result)

                    avg_score = (
                        gen_result.faithfulness + gen_result.relevance + gen_result.completeness
                    ) / 3
                    print(
                        f"  [{i:2d}/{len(gen_queries)}] "
                        f"F={gen_result.faithfulness} R={gen_result.relevance} "
                        f"C={gen_result.completeness} Avg={avg_score:.1f}  "
                        f"({latency:.1f}s) | {query[:50]}"
                    )
                except Exception as e:
                    print(f"  ✗ Judging failed for: {query[:50]} ({e})")

            # Aggregate generation metrics
            if generation_results:
                n = len(generation_results)
                generation_metrics = {
                    "count": n,
                    "avg_faithfulness": sum(r.faithfulness for r in generation_results) / n,
                    "avg_relevance": sum(r.relevance for r in generation_results) / n,
                    "avg_completeness": sum(r.completeness for r in generation_results) / n,
                    "avg_overall": sum(
                        (r.faithfulness + r.relevance + r.completeness) / 3
                        for r in generation_results
                    ) / n,
                    "avg_latency": sum(r.latency_seconds for r in generation_results) / n,
                }

                print(f"\n{'─' * 60}")
                print(f"📊 Generation Results:")
                print(f"   Avg Faithfulness: {generation_metrics['avg_faithfulness']:.2f} / 5")
                print(f"   Avg Relevance:    {generation_metrics['avg_relevance']:.2f} / 5")
                print(f"   Avg Completeness: {generation_metrics['avg_completeness']:.2f} / 5")
                print(f"   Avg Overall:      {generation_metrics['avg_overall']:.2f} / 5")
                print(f"   Avg Latency:      {generation_metrics['avg_latency']:.1f}s")

    # ---- Compile final report ----
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "csv_path": csv_path,
            "num_trials": len(documents),
            "top_k": top_k,
            "model": model,
            "retriever_type": retriever_type if (retriever_type == "voyage" and voyage_api_key) else "tfidf",
            "num_eval_queries": len(EVAL_QUERIES),
        },
        "retrieval": {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "avg_precision": avg_precision,
            "per_category": category_metrics,
            "details": [asdict(r) for r in retrieval_results],
        },
        "generation": {
            "metrics": generation_metrics,
            "details": [asdict(r) for r in generation_results],
        },
    }

    print(f"\n{'=' * 60}")
    print(f"✅ Evaluation complete!")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument(
        "--csv", default=None,
        help="Path to the clinical trials CSV. Defaults to the full dataset."
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only evaluate retrieval, skip generation (no API key needed)"
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514",
        help="Model for generation and judging"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Defaults to backend/data/eval_results_YYYY-MM-DD.json"
    )
    parser.add_argument(
        "--retriever", default="voyage", choices=["voyage", "tfidf"],
        help="Retriever type: 'voyage' (dense embeddings) or 'tfidf' (sparse). Default: voyage"
    )
    args = parser.parse_args()

    # Default CSV path
    if args.csv:
        csv_path = args.csv
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        # Try full dataset first, fall back to any available CSV
        full_path = os.path.join(data_dir, "breast_cancer_trials_full_2026-04-14.csv")
        if os.path.exists(full_path):
            csv_path = full_path
        else:
            import glob
            csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
            if csvs:
                csv_path = csvs[-1]
            else:
                print("✗ No CSV files found in data/")
                return

    # Voyage API key
    voyage_api_key = os.getenv("VOYAGE_API_KEY", "")

    # Run evaluation
    report = asyncio.run(run_evaluation(
        csv_path=csv_path,
        top_k=args.top_k,
        retrieval_only=args.retrieval_only,
        model=args.model,
        retriever_type=args.retriever,
        voyage_api_key=voyage_api_key,
    ))

    # Save report
    if args.output:
        output_path = args.output
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = os.path.join(data_dir, f"eval_results_{date_str}.json")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()
