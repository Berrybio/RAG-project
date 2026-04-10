import argparse
import os

from part4_combine import ClinicalTrialRAG

# ---------------------------------------------------------------------------
# 6. ENVIRONMENT DETECTION & MAIN
# ---------------------------------------------------------------------------

def _is_notebook() -> bool:
    """Detect if we're running inside a Jupyter/Colab notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":   # Jupyter / Colab
            return True
        if "google.colab" in str(get_ipython()):  # fallback Colab check
            return True
    except (ImportError, AttributeError):
        pass
    return False


def _locate_csv() -> str:
    """Find the CSV file — handles both script and notebook contexts."""
    # 1. Check current working directory first (common in Colab)
    cwd_path = os.path.join(os.getcwd(), "Breast_Cancer-RECRUITING-phase2-625.csv")
    if os.path.exists(cwd_path):
        return cwd_path

    # 2. Check script directory (when run as python script)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "Breast_Cancer-RECRUITING-phase2-625.csv")
        if os.path.exists(script_path):
            return script_path
    except NameError:
        pass  # __file__ not defined in notebooks

    # 3. Check /content/ (Colab's default upload directory)
    colab_path = "/content/Breast_Cancer-RECRUITING-phase2-625.csv"
    if os.path.exists(colab_path):
        return colab_path

    raise FileNotFoundError(
        "CSV file 'Breast_Cancer-RECRUITING-phase2-625.csv' not found.\n"
        "Please place it in the current directory, or in Colab upload it to /content/"
    )


def run(
    retriever_type: str = "tfidf",
    model: str = "claude-opus-4-6",
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_dir: str | None = None,
    csv_path: str | None = None,
    query: str | None = None,
    top_k: int = 5,
) -> ClinicalTrialRAG:
    """
    Notebook-friendly entry point.  Call this directly in Colab / Jupyter:

        # In a Colab cell:
        rag = run(retriever_type="tfidf")

        # Then ask questions:
        rag.ask("What trials are studying immunotherapy for TNBC?")
        rag.ask("Are there trials in New York?")

    Args:
        retriever_type: "tfidf", "chroma", or "hybrid"
        model:          Claude model ID for generation
        embedding_model: sentence-transformer model (for chroma/hybrid)
        persist_dir:    directory to persist ChromaDB index on disk
        csv_path:       path to CSV (auto-detected if None)
        query:          optional query to run immediately after init
        top_k:          number of trials to retrieve per query

    Returns:
        The initialized ClinicalTrialRAG instance (reuse it for more queries)
    """
    if csv_path is None:
        csv_path = _locate_csv()

    rag = ClinicalTrialRAG(
        csv_path,
        retriever_type=retriever_type,
        model=model,
        persist_directory=persist_dir,
        embedding_model=embedding_model,
    )

    if query:
        rag.ask(query, top_k=top_k)

    return rag


def main():
    """CLI entry point — only uses argparse when running as a script."""

    # --- Notebook mode: skip argparse entirely ---
    if _is_notebook():
        print("Notebook environment detected — use run() or ClinicalTrialRAG directly.")
        print()
        print("Quick start (paste in a cell):")
        print('  rag = run(retriever_type="tfidf")')
        print('  rag.ask("What trials are studying immunotherapy for TNBC?")')
        print()
        print("Options:")
        print('  rag = run(retriever_type="chroma")   # semantic embeddings')
        print('  rag = run(retriever_type="hybrid")   # best quality')
        return

    # --- CLI mode: parse arguments normally ---
    parser = argparse.ArgumentParser(
        description="RAG system for breast cancer clinical trials"
    )
    parser.add_argument(
        "--retriever",
        choices=["tfidf", "chroma", "hybrid"],
        default="tfidf",
        help=(
            "Retrieval method: "
            "'tfidf' (fast, keyword-based), "
            "'chroma' (semantic embeddings via sentence-transformers), "
            "'hybrid' (best quality, combines both). "
            "Default: tfidf"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help=(
            "Sentence-transformer model for dense embeddings. "
            "Default: all-MiniLM-L6-v2. "
            "For higher quality try: all-mpnet-base-v2 (~420 MB)"
        ),
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help=(
            "Directory to persist the ChromaDB index on disk. "
            "Avoids re-embedding on subsequent runs. "
            "Only used with --retriever chroma or hybrid."
        ),
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Claude model ID for generation. Default: claude-opus-4-6",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to the clinical trials CSV file. Auto-detected if not provided.",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv else _locate_csv()

    rag = ClinicalTrialRAG(
        csv_path,
        retriever_type=args.retriever,
        model=args.model,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
    )

    # Example queries
    example_queries = [
        "What clinical trials are available for triple-negative breast cancer?",
        "Are there any immunotherapy trials for HER2-positive breast cancer?",
        "Find trials that accept patients over 65 years old",
        "What trials are using CDK4/6 inhibitors?",
        "Are there clinical trials available in New York?",
    ]

    print("=" * 60)
    print(f"CLINICAL TRIAL RAG  (retriever: {args.retriever})")
    print("=" * 60)
    print("\nExample queries you can ask:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    print()

    # Interactive loop
    while True:
        print("\n" + "=" * 60)
        user_input = input(
            "Enter your question (or 'quit' to exit, '1'-'5' for examples): "
        ).strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input in ("1", "2", "3", "4", "5"):
            user_input = example_queries[int(user_input) - 1]

        if not user_input:
            continue

        print()
        rag.ask(user_input, top_k=5)


if __name__ == "__main__":
    main()
