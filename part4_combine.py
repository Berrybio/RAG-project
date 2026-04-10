import anthropic

from part1 import load_clinical_trials, build_documents
from part2_retriever_tfidf import BaseRetriever, TFIDFRetriever
from part3_generation import generate_answer
from rag_clinical_trials import ChromaRetriever, HybridRetriever

# 5. RAG PIPELINE — PUTTING IT ALL TOGETHER
# ---------------------------------------------------------------------------

class ClinicalTrialRAG:
    """
    End-to-end RAG pipeline for clinical trial Q&A.

    Usage:
        # TF-IDF (lightweight, default)
        rag = ClinicalTrialRAG("data.csv", retriever_type="tfidf")

        # ChromaDB dense embeddings (better semantic understanding)
        rag = ClinicalTrialRAG("data.csv", retriever_type="chroma")

        # Hybrid (best quality — combines both)
        rag = ClinicalTrialRAG("data.csv", retriever_type="hybrid")

        answer = rag.ask("What trials study immunotherapy for TNBC?")
    """

    def __init__(
        self,
        csv_path: str,
        retriever_type: str = "tfidf",
        model: str = "claude-opus-4-6",
        persist_directory: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        print("Loading clinical trial data...")
        self.df = load_clinical_trials(csv_path)
        print(f"Loaded {len(self.df)} trials.\n")

        print("Building document representations...")
        self.documents = build_documents(self.df)

        print(f"Creating retriever (mode: {retriever_type})...")
        self.retriever = self._build_retriever(
            retriever_type, persist_directory, embedding_model
        )

        print("\nInitializing Claude client...")
        self.client = anthropic.Anthropic()
        self.model = model

        print("RAG pipeline ready!\n")

    def _build_retriever(
        self,
        retriever_type: str,
        persist_directory: str | None,
        embedding_model: str,
    ) -> BaseRetriever:
        """Factory method to create the chosen retriever."""
        if retriever_type == "tfidf":
            return TFIDFRetriever(self.documents)

        elif retriever_type == "chroma":
            return ChromaRetriever(
                self.documents,
                persist_directory=persist_directory,
                embedding_model=embedding_model,
            )

        elif retriever_type == "hybrid":
            return HybridRetriever(
                self.documents,
                persist_directory=persist_directory,
                embedding_model=embedding_model,
            )

        else:
            raise ValueError(
                f"Unknown retriever_type '{retriever_type}'. "
                "Choose from: tfidf, chroma, hybrid"
            )

    def ask(self, query: str, top_k: int = 5) -> str:
        """
        Ask a question about clinical trials.

        Args:
            query:  Natural language question
            top_k:  Number of trials to retrieve for context

        Returns:
            The generated answer string
        """
        print(f"Query: {query}")
        print(f"Retrieving top {top_k} relevant trials...\n")

        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        if not retrieved:
            return "No relevant clinical trials found for your query."

        print(f"Found {len(retrieved)} relevant trials:")
        for doc in retrieved:
            title = doc['metadata']['title']
            if len(title) > 80:
                title = title[:80] + "..."
            print(f"  - {doc['doc_id']}: {title} (score: {doc['score']:.3f})")
        print()

        # Step 2: Generate
        print("Generating answer...\n")
        print("-" * 60)
        answer = generate_answer(self.client, query, retrieved, self.model)
        print("-" * 60)

        return answer

