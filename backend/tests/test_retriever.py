def test_retriever_returns_results(sample_retriever):
    results = sample_retriever.retrieve("CDK4/6 inhibitor", top_k=3)
    assert len(results) > 0
    assert results[0]["doc_id"] == "NCT00000001"
    assert results[0]["score"] > 0


def test_retriever_ranking(sample_retriever):
    results = sample_retriever.retrieve("immunotherapy TNBC", top_k=3)
    assert len(results) > 0
    # TNBC immunotherapy trial should rank first
    assert results[0]["doc_id"] == "NCT00000002"


def test_retriever_scores_descending(sample_retriever):
    results = sample_retriever.retrieve("breast cancer", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_top_k_limit(sample_retriever):
    results = sample_retriever.retrieve("breast cancer", top_k=1)
    assert len(results) == 1


def test_retriever_result_structure(sample_retriever):
    results = sample_retriever.retrieve("HER2", top_k=1)
    assert len(results) > 0
    result = results[0]
    assert "doc_id" in result
    assert "text" in result
    assert "metadata" in result
    assert "score" in result
    assert isinstance(result["score"], float)
