# milestone3.py

from typing import Any, Dict, Tuple
import json
from llm_layer import main_llm_call
from neo4j_client import neo4j_client
from semantic_parser import parse_user_query
from queries import (
    template_1_hotels_in_city,
    template_2_hotels_in_country,
    template_3_hotels_by_rating_option,
    template_4_latest_reviews_for_hotel,
    template_5_popular_hotels_for_nationality,
    template_6_visa_requirements,
    template_7_hotels_for_traveller_type_and_age,
    template_8_hotels_by_aspect_scores,
    template_9_visa_free_destinations,
    template_10_compare_hotels,
)
from vector_embeddings import run_vector_search_model1, run_vector_search_model2


def _format_cypher_answer_for_llm(results) -> str:
    if not results:
        return ""

    lines = []
    for row in results:
        h = row.get("h") or {}
        city = row.get("city") or {}
        country = row.get("country") or {}

        hotel_name = h.get("name", "Unknown")
        city_name = city.get("name", "Unknown")
        country_name = country.get("name", "Unknown")

        stars = h.get("star_rating")
        avg_review = h.get("average_reviews_score")

        cleanliness = row.get("cleanliness", h.get("cleanliness"))
        comfort = row.get("comfort", h.get("comfort"))
        facilities = row.get("facilities", h.get("facilities"))

        parts = [
            f"- {hotel_name}",
            f"City: {city_name}",
            f"Country: {country_name}",
            f"Stars: {stars}",
            f"Avg Review: {avg_review}",
        ]

        if cleanliness is not None:
            parts.append(f"Cleanliness: {cleanliness}")
        if comfort is not None:
            parts.append(f"Comfort: {comfort}")
        if facilities is not None:
            parts.append(f"Facilities: {facilities}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)


def _format_reviews_answer_for_llm(results) -> str:

    if not results:
        return "Reviews: (none retrieved)"

    lines = ["Reviews (latest):"]
    for row in results:
        h = row.get("h") or {}
        r = row.get("r") or {}
        t = row.get("t") or {}

        lines.append(
            f"- Hotel: {h.get('name','Unknown')}"
            f" | Traveller type: {t.get('type','Unknown')}"
            f" | Score: {r.get('score_overall','Unknown')}"
            f" | Review: {str(r.get('text',''))[:250]}"
        )
    return "\n".join(lines)


def _format_visa_answer_for_llm(results, mode: str) -> str:

    if not results:
        return "Visa info: (none retrieved)"

    if mode == "requirements":
        lines = ["Visa requirements:"]
        for row in results:
            lines.append(f"- Visa type required: {row.get('visa_type','Unknown')}")
        return "\n".join(lines)

    if mode == "visa_free":
        lines = ["Visa-free destinations:"]
        for row in results:
            to_country = row.get("to") or {}
            lines.append(
                f"- {to_country.get('name','Unknown')} ({row.get('visa_type','no_visa')})"
            )
        return "\n".join(lines)

    return str(results)


def _deduce_rating_option(min_star: float | None, max_star: float | None) -> str | None:

    if min_star is not None and max_star is None:
        return f"{min_star:g}+"

    if min_star is None and max_star is not None:
        return f"-{max_star:g}"

    if min_star is not None and max_star is not None:
        if min_star == max_star:
            return f"{min_star:g}"
        if min_star < max_star:
            return f"{min_star:g}-{max_star:g}"

    return None



def choose_hotel_query(parsed: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:

    cities = parsed["cities"]  # list of lowercase city names
    countries = parsed["countries"]  # list of lowercase country names
    trav_types = parsed["traveller_types"]
    min_star = parsed["min_star_rating"]
    max_star = parsed["max_star_rating"]
    min_review = parsed["min_review_score"]
    preferred_aspects = parsed["preferred_aspects"]

    primary_city = cities[0] if cities else None
    primary_country = countries[0] if countries else None


    if preferred_aspects:
        tpl = template_8_hotels_by_aspect_scores(
            aspects=preferred_aspects,
            city=primary_city,
            country=primary_country,
            limit=20,
        )
        return tpl["query"], tpl["params"]

    if trav_types:
        traveller_type = trav_types[0]  # take first detected type
        tpl = template_7_hotels_for_traveller_type_and_age(
            traveller_type=traveller_type,
            min_age=None,
            max_age=None,
            city=primary_city,
            country=primary_country,
            limit=20,
        )
        return tpl["query"], tpl["params"]

    if primary_city:
        tpl = template_1_hotels_in_city(
            city=primary_city,
            min_star=min_star,
            max_star=max_star,
            min_review=min_review,
            limit=20,
        )
        return tpl["query"], tpl["params"]

    if primary_country:
        # If user gave multiple cities in same country, we can pass them all.
        extra_cities = cities if len(cities) > 1 else None
        tpl = template_2_hotels_in_country(
            country=primary_country,
            cities=extra_cities,
            min_star=min_star,
            max_star=max_star,
            min_review=min_review,
            limit=20,
        )
        return tpl["query"], tpl["params"]

    # 4. No clear location. Use rating-based query (template_3) as fallback.
    rating_option = _deduce_rating_option(min_star, max_star) or "4+"
    tpl = template_3_hotels_by_rating_option(
        rating_option=rating_option,
        city=None,
        country=None,
        limit=20,
    )
    return tpl["query"], tpl["params"]


def handle_hotel_recommendation_intent(parsed: Dict[str, Any]):
    query, params = choose_hotel_query(parsed)
    print("Using hotel RECOMMENDATION query with params:", params)
    results = neo4j_client.run(query, params)

    cypher_answer = _format_cypher_answer_for_llm(results)

    return cypher_answer , query, params, results


def handle_hotel_search_intent(parsed: Dict[str, Any]):
    cities = parsed["cities"]
    countries = parsed["countries"]
    min_star = parsed["min_star_rating"]
    max_star = parsed["max_star_rating"]
    min_review = parsed["min_review_score"]

    if cities:
        tpl = template_1_hotels_in_city(
            city=cities[0],
            min_star=min_star,
            max_star=max_star,
            min_review=min_review,
            limit=20,
        )
    elif countries:
        tpl = template_2_hotels_in_country(
            country=countries[0],
            cities=None,
            min_star=min_star,
            max_star=max_star,
            min_review=min_review,
            limit=20,
        )
    else:
        rating_option = _deduce_rating_option(min_star, max_star) or "4+"
        tpl = template_3_hotels_by_rating_option(
            rating_option=rating_option,
            city=None,
            country=None,
            limit=20,
        )

    query, params = tpl["query"], tpl["params"]
    results = neo4j_client.run(query, params)

    cypher_answer = _format_cypher_answer_for_llm(results)

    return cypher_answer , query,params , results


def handle_hotel_reviews_intent(parsed: Dict[str, Any]):
    hotels = parsed.get("hotels", [])
    query = ""
    params: Dict[str, Any] = {}
    results = []
    cypher_answer = ""

    # 1. If user mentioned multiple hotels, compare them.
    if len(hotels) >= 2:
        print(f"Comparing hotels: {hotels}")
        tpl = template_10_compare_hotels(hotel_names=hotels)
        query, params = tpl["query"], tpl["params"]
        results = neo4j_client.run(query, params)

        if not results:
            cypher_answer = ""
        else:
            lines = []
            for row in results:
                h = row.get("h") or {}
                city = row.get("city") or {}
                country = row.get("country") or {}
                lines.append(
                    f"- {h.get('name','Unknown')}"
                    f" | City: {city.get('name','Unknown')}"
                    f" | Country: {country.get('name','Unknown')}"
                    f" | Stars: {row.get('star_rating')}"
                    f" | Avg Review: {row.get('avg_overall')}"
                )
            cypher_answer = "\n".join(lines)

        return cypher_answer,query, params, results

    # 2. If only one hotel name, show its latest reviews.
    if len(hotels) == 1:
        tpl = template_4_latest_reviews_for_hotel(hotel_name=hotels[0], limit=10)
        query, params = tpl["query"], tpl["params"]
        results = neo4j_client.run(query, params)

        cypher_answer = _format_reviews_answer_for_llm(results)

        return cypher_answer,query, params, results

    msg = "hotel_reviews intent but no hotel names detected."
    cypher_answer = msg
    return cypher_answer,query, params, results


def handle_visa_search_intent(parsed: Dict[str, Any]):
    countries = parsed["countries"]

    # requirements: 2 countries
    if len(countries) >= 2:
        from_country = countries[1]
        to_country = countries[0]
        tpl = template_6_visa_requirements(
            from_country=from_country, to_country=to_country
        )
        query, params = tpl["query"], tpl["params"]
        results = neo4j_client.run(query, params)

        cypher_answer = _format_visa_answer_for_llm(results, mode="requirements")

        return cypher_answer,query, params, results

    # visa-free: 1 country
    if len(countries) == 1:
        from_country = countries[0]
        tpl = template_9_visa_free_destinations(from_country=from_country)
        query, params = tpl["query"], tpl["params"]
        results = neo4j_client.run(query, params)

        cypher_answer = _format_visa_answer_for_llm(results, mode="visa_free")

        return cypher_answer,query, params, results

    msg = "visa_search intent but no countries specified."
    print(msg)
    return msg

def _safe_json(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def run_graph_rag(
    user_text: str,
    llm_model: str = "mistral",
    retrieval_mode: str = "baseline",
) -> Dict[str, Any]:

    parsed = parse_user_query(user_text)
    parsed["raw_text"] = user_text
    intent = parsed.get("intent", "unknown")

    cypher_query = ""
    cypher_params: Dict[str, Any] = {}
    results = []

    if intent == "hotel_recommendation":

        cypher_answer,cypher_query, cypher_params, results = handle_hotel_recommendation_intent(parsed)

    elif intent == "hotel_search":
        cypher_answer,cypher_query, cypher_params, results = handle_hotel_search_intent(parsed)

    elif intent == "hotel_reviews":
        cypher_answer,cypher_query,cypher_params,results = handle_hotel_reviews_intent(parsed)

    elif intent == "visa_requirements":
        cypher_answer, cypher_query, cypher_params,results = handle_visa_search_intent(parsed)

    else:
        cypher_query, cypher_params = choose_hotel_query(parsed)
        results = neo4j_client.run(cypher_query, cypher_params)
        cypher_answer = _format_cypher_answer_for_llm(results)

    embeddings_answer_model1 = run_vector_search_model1(user_text, top_k=10)
    embeddings_answer_model2 = run_vector_search_model2(user_text, top_k=10)

    if retrieval_mode == "baseline":
        embeddings_answer_model1 = ""
        embeddings_answer_model2 = ""

    elif retrieval_mode == "embeddings":
        cypher_answer = ""

    final_answer = main_llm_call(
        cypher_answer=cypher_answer,
        embeddings_model1=embeddings_answer_model1,
        embeddings_model2=embeddings_answer_model2,
        llm_model=llm_model,
        user_query=user_text,
    )

    return {
        "intent": intent,
        "parsed": _safe_json(parsed),
        "retrieval_mode": retrieval_mode,
        "llm_model": llm_model,
        "cypher_query": cypher_query,
        "cypher_params": _safe_json(cypher_params),
        "raw_rows": _safe_json(results),
        "cypher_answer": cypher_answer,
        "embeddings_model1": embeddings_answer_model1,
        "embeddings_model2": embeddings_answer_model2,
        "final_answer": final_answer,
    }

