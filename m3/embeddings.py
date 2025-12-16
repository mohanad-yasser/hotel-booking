import pandas as pd
from sentence_transformers import SentenceTransformer
from neo4j_client import neo4j_client

MODEL_1_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384 dims
MODEL_2_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768 dims

PROP_M1 = "textEmbedding_m1"
PROP_M2 = "textEmbedding_m2"



def load_hotels_from_neo4j() -> pd.DataFrame:
    """
    Loads hotels + city + country + numeric hotel features from Neo4j.
    Adjust property names if your graph differs.
    """
    cypher = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
    OPTIONAL MATCH (c)-[:LOCATED_IN]->(co:Country)
    RETURN h.hotel_id AS hotel_id,
           h.name AS hotel_name,
           c.name AS city,
           co.name AS country,
           h.star_rating AS star_rating,
           h.cleanliness_base AS cleanliness_base,
           h.comfort_base AS comfort_base,
           h.facilities_base AS facilities_base
          
    """
    rows = neo4j_client.run(cypher, {})
    return pd.DataFrame(rows)


# 2) Build one text doc per hotel 
def build_hotel_docs(hotels_df: pd.DataFrame) -> dict[int, str]:
    if hotels_df.empty:
        raise RuntimeError("No hotels returned from Neo4j. Check your Hotel/City relationships and properties.")

    hotels_df["hotel_id"] = hotels_df["hotel_id"].astype(int)

    docs: dict[int, str] = {}

    for _, h in hotels_df.iterrows():
        hid = int(h["hotel_id"])

        doc = (
            f"{h.get('hotel_name','')} located in {h.get('city','')}, {h.get('country','')}. "
            f"Stars {h.get('star_rating','')}. "
            f"Cleanliness {h.get('cleanliness_base','')}. "
            f"Comfort {h.get('comfort_base','')}. "
            f"Facilities {h.get('facilities_base','')}. "
            f"Location {h.get('location_base','')}. "
            f"Staff {h.get('staff_base','')}. "
            f"Value for money {h.get('value_for_money_base','')}."
        )

        docs[hid] = doc

    return docs


# 3) Write embeddings back to Neo4j

def write_embeddings(prop_name: str, hotel_ids: list[int], embeddings_np):
    query = f"""
    UNWIND $rows AS row
    MATCH (h:Hotel {{hotel_id: row.hotel_id}})
    SET h.{prop_name} = row.embedding
    """
    rows = []
    for i, hid in enumerate(hotel_ids):
        rows.append({"hotel_id": hid, "embedding": embeddings_np[i].astype(float).tolist()})
    neo4j_client.run(query, {"rows": rows})


# 4) Create vector indexes (2 models)
def create_vector_indexes():
    index_m1_query = """
    CREATE VECTOR INDEX hotelTextIndex_m1 IF NOT EXISTS
    FOR (h:Hotel)
    ON (h.textEmbedding_m1)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
      }
    }
    """

    index_m2_query = """
    CREATE VECTOR INDEX hotelTextIndex_m2 IF NOT EXISTS
    FOR (h:Hotel)
    ON (h.textEmbedding_m2)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
      }
    }
    """

    neo4j_client.run(index_m1_query)
    neo4j_client.run(index_m2_query)
    print("Vector indexes created (or already existed).")



# 5) Main
def main():
    hotels_df = load_hotels_from_neo4j()

    docs_by_hotel = build_hotel_docs(hotels_df)
    hotel_ids = list(docs_by_hotel.keys())
    docs = list(docs_by_hotel.values())

    # Model 1
    print(f"Loading model 1: {MODEL_1_NAME}")
    embedder1 = SentenceTransformer(MODEL_1_NAME)
    print("Encoding hotels with model 1...")
    emb1 = embedder1.encode(docs, convert_to_numpy=True)
    print("Model 1 embeddings shape:", emb1.shape)
    write_embeddings(PROP_M1, hotel_ids, emb1)

    # Model 2
    print(f"Loading model 2: {MODEL_2_NAME}")
    embedder2 = SentenceTransformer(MODEL_2_NAME)
    print("Encoding hotels with model 2...")
    emb2 = embedder2.encode(docs, convert_to_numpy=True)
    print("Model 2 embeddings shape:", emb2.shape)
    write_embeddings(PROP_M2, hotel_ids, emb2)

    create_vector_indexes()

    print("Done. Stored:")
    print(f"- Hotel.{PROP_M1} (model 1)")
    print(f"- Hotel.{PROP_M2} (model 2)")


if __name__ == "__main__":
    main()
