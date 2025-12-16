# benchmark_runner.py
import json
from milestone3 import run_graph_rag
from llm_layer import run_llm_benchmark

def build_test_cases():
    # Cover your intents/themes
    queries = [
        ("TC01", "Hotels in Paris", ["paris", "hotel"]),
        ("TC02", "Show me the latest reviews for The Golden Oasis", ["golden oasis", "review"]),
        ("TC03", "Can i travel from Egypt to Singapore without a visa?", ["visa", "singapore"]),
    ]

    test_cases = []
    for tc_id, user_query, expected_keywords  in queries:
        out = run_graph_rag(
            user_text=user_query,
            llm_model="deepseek",       # doesn't matter for context snapshot
            retrieval_mode="hybrid",    # keeps cypher + embeddings
        )

        # IMPORTANT: your benchmark expects ONE embeddings string,
        # so choose which embedding output you want to evaluate.
        # Here we evaluate model1 embeddings (MiniLM) as the prototype.
        embeddings_answer = out.get("embeddings_model1", "") or ""

        test_cases.append({
            "id": tc_id,
            "user_query": user_query,
            "cypher_answer": out.get("cypher_answer", "") or "",
            "embeddings_answer": embeddings_answer,
            "expected_keywords": expected_keywords,
        })

    with open("test_cases.json", "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    return test_cases

if __name__ == "__main__":
    cases = build_test_cases()
    run_llm_benchmark(cases)
