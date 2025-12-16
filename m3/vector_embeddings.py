from sentence_transformers import SentenceTransformer
from neo4j_client import neo4j_client

embedder1 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # 384
embedder2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # 768

def format_vector_results(rows, title: str) -> str:
    if not rows:
        return f"{title}\nNo results."

    lines = [title]
    for row in rows:
        h = row.get("h") or {}
        city = row.get("city") or {}
        country = row.get("country") or {}
        score = row.get("score")

        lines.append(
            f"- {h.get('name','Unknown')} | "
            f"{city.get('name','Unknown')}, {country.get('name','Unknown')} | "
            f"sim={score:.4f}"
        )
    return "\n".join(lines)

def embed_user_text_model1(user_text: str) -> list[float]:
    text = (user_text or "").strip()
    if not text:
        return []
    return embedder1.encode(text).tolist()


def embed_user_text_model2(user_text: str) -> list[float]:
    text = (user_text or "").strip()
    if not text:
        return []
    return embedder2.encode(text).tolist()


def template_vector_search_hotels(index_name: str, limit: int = 10):
    query = """
    CALL db.index.vector.queryNodes($index_name, $limit, $query_embedding)
    YIELD node AS h, score
    OPTIONAL MATCH (h)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
    RETURN h, city, country, score
    ORDER BY score DESC
    """
    params = {"index_name": index_name, "limit": limit}
    return {"query": query, "params": params}


def run_vector_search_model1(user_text: str, top_k: int = 10):
    """
    Runs semantic search using Model 1 (MiniLM) and returns Neo4j rows.
    """
    v1 = embed_user_text_model1(user_text)
    if not v1:
        return []

    tpl = template_vector_search_hotels(index_name="hotelTextIndex_m1", limit=top_k)
    params = tpl["params"]
    params["query_embedding"] = v1
    answer=neo4j_client.run(tpl["query"], params)
    query_answer = format_vector_results(answer, "Embeddings Results (Model 1):")

    return query_answer


def run_vector_search_model2(user_text: str, top_k: int = 10):
    """
    Runs semantic search using Model 2 (MPNet) and returns Neo4j rows.
    """
    v2 = embed_user_text_model2(user_text)
    if not v2:
        return []

    tpl = template_vector_search_hotels(index_name="hotelTextIndex_m2", limit=top_k)
    params = tpl["params"]
    params["query_embedding"] = v2
    answer=neo4j_client.run(tpl["query"], params)
    query_answer = format_vector_results(answer, "Embeddings Results (Model 2):")

    return query_answer
