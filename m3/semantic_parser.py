from typing import Any, Dict, List
import json
import re

import spacy

from neo4j_client import neo4j_client
from hf_client import call_hf_llm
from spacy.pipeline import EntityRuler


nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    # COUPLE patterns
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "couple"}]},
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "me"}, {"LOWER": "and"}, {"LOWER": "my"}, {"LOWER": {"IN": ["wife", "husband", "partner", "spouse"]}}]},
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "my"}, {"LOWER": {"IN": ["wife", "husband", "partner", "spouse"]}}, {"LOWER": "and"}, {"LOWER": "i"}]},

    # FAMILY patterns
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "family"}]},
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "with"}, {"LOWER": "my"}, {"LOWER": {"IN": ["kids", "children"]}}]},
    {
    "label": "TRAVELLER_TYPE",
    "pattern": [
        {"LOWER": {"IN": ["my", "with"]}},
        {"LOWER": {"IN": ["wife", "husband", "spouse", "partner"]}},
        {"LOWER": "and"},
        {"LOWER": {"IN": ["children", "kids"]}}
    ]
    },

    # BUSINESS patterns
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "business"}]},
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "business"}, {"LOWER": "trip"}]},
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "work"}, {"LOWER": "trip"}]},

    # SOLO patterns
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "solo"}]},
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "alone"}]},
    {"label": "TRAVELLER_TYPE", "pattern": [{"LOWER": "by"}, {"LOWER": "myself"}]},
]

ruler.add_patterns(patterns)


def _load_cities() -> List[str]:
    res = neo4j_client.run("MATCH (c:City) RETURN toLower(c.name) AS name")
    return [row["name"] for row in res]


def _load_countries() -> List[str]:
    res = neo4j_client.run("MATCH (c:Country) RETURN toLower(c.name) AS name")
    return [row["name"] for row in res]


def _load_hotels() -> Dict[str, str]:
    """
    Load hotel names from the KG.
    Key   = lowercase name (for matching)
    Value = original name (for Neo4j queries)
    """
    res = neo4j_client.run(
        "MATCH (h:Hotel) RETURN toLower(h.name) AS lname, h.name AS name"
    )
    return {row["lname"]: row["name"] for row in res}



CITY_LIST = _load_cities()
COUNTRY_LIST = _load_countries()
HOTEL_MAP = _load_hotels()          # dict
HOTEL_LIST = list(HOTEL_MAP.keys())  # lowercase list for matching





# -----------------------------
# LLM call: extract ONLY the intent
# -----------------------------
def _llm_extract(user_text: str):
    """
    Ask the LLM to classify the user intent ONLY.
    NER will handle all variables (cities, countries, hotels, etc.).
    """

    prompt = f"""
You are a strict JSON generator for a hotel assistant.
Read the user query and output ONLY a JSON object with this shape:

{{
  "intent": "<one of ['hotel_search', 'hotel_recommendation', 'hotel_reviews', 'visa_requirements',"popular_hotels_for_nationality", 'unknown_intent']>"
}}

Definitions:
- "hotel_search":
    The user is searching for hotels in a location, maybe with simple filters
    like star rating or review score, but not clearly asking for personalized
    recommendations. look for words like "find", "show me", "looking for", "search"
    Examples: "Show me hotels in Cairo", "Find 3-star hotels in France"

- "hotel_recommendation":
    The user asks for suggestions or the best options for them.
    Look for words like "recommend", "suggest", "which hotel should I stay in",
    "best hotel for me", etc.
    Examples: "Can you recommend a hotel in Berlin for me?", "What is the best hotel in Rome?"

- "hotel_reviews":
    The user wants reviews,or opinions about hotels or stays.
    Look for words like "reviews", "feedback", "what do people say",
    "are the reviews good", etc.
    Examples: "Show me reviews for hotels in Dubai", "What do people say about Hilton Cairo?", "compare between Marriott and Four Seasons"

- "visa_requirements":
    The user asks about visa rules between two countries.
    Look for words like "visa", "visa requirements", "do I need a visa",
    "visa free", etc.
    Examples: "Do Egyptians need a visa for Turkey?", "Is it visa free to travel from France to Spain?"

- "popular_hotels_for_nationality":
    The user asks about hotels preferred or popular among travellers from a
    specific country or nationality.
    Look for phrases like:
      "travellers from <country>"
      "tourists from <country>"
      "<nationality> travellers"
      "popular with <country> visitors"
    Examples:
      "Most popular hotels in Dubai for travellers from Egypt"
      "Hotels in France liked by German tourists"
      "Which hotels do Egyptians prefer in Istanbul?"

- "unknown_intent":
    If the user query does not match any of the above intents.

Choose the single best intent according to these rules.

USER_QUERY: \"\"\"{user_text}\"\"\"

Return ONLY a valid JSON object. No extra text.
"""

    raw = call_hf_llm(prompt)

    try:
        data = json.loads(raw)
    except Exception:
        data = {"intent": "unknown_intent"}

    return data



# -----------------------------
# spaCy NER for cities/countries
# -----------------------------
def ner_locations(user_text: str):
    doc = nlp(user_text)
    cities_found = set()
    countries_found = set()

    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            name = ent.text.lower()
            if name in CITY_LIST:
                cities_found.add(name)
            if name in COUNTRY_LIST:
                countries_found.add(name)

    return list(cities_found), list(countries_found)


# -----------------------------
# Hotel name extraction (from KG)
# -----------------------------
def ner_hotels(user_text: str):
    doc = nlp(user_text)
    hotels_found = set()
    text_lower = user_text.lower()

    # 1. Use spaCy entities
    for ent in doc.ents:
        candidate = ent.text.lower().strip()

        if candidate in HOTEL_MAP:
            # ADD ORIGINAL KG NAME
            hotels_found.add(HOTEL_MAP[candidate])

    # 2. Fallback: substring match
    for lname, original_name in HOTEL_MAP.items():
        if lname in text_lower:
            hotels_found.add(original_name)

    return list(hotels_found)




# -----------------------------
# Numeric extraction helpers
# -----------------------------
def _to_float(x: Any):
    try:
        return float(x)
    except Exception:
        return None



def extract_star_rating(user_text: str):
    text = user_text.lower()

    min_star: float | None = None
    max_star: float | None = None

    # -------------------------
    # 1) Range patterns first
    # -------------------------
    # between 3 and 5 stars
    m = re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*stars?\b", text)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        min_star, max_star = (a, b) if a <= b else (b, a)
        return {"min_star_rating": min_star, "max_star_rating": max_star}

    # 3 to 5 stars
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:to|and)\s*(\d+(?:\.\d+)?)\s*stars?\b", text)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        min_star, max_star = (a, b) if a <= b else (b, a)
        return {"min_star_rating": min_star, "max_star_rating": max_star}

    # 3-4 stars / 3 - 4 star
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*stars?\b", text)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        min_star, max_star = (a, b) if a <= b else (b, a)
        return {"min_star_rating": min_star, "max_star_rating": max_star}

    # -------------------------
    # 2) "4+ stars"
    # -------------------------
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*\+\s*stars?\b", text)
    if m:
        min_star = float(m.group(1))
        return {"min_star_rating": min_star, "max_star_rating": None}

    # -------------------------
    # 3) Single-number patterns with operators
    # -------------------------
    # We'll capture a star number and look around it for operator words.
    # Example: "below 4 stars", "at least 4 stars", "4 stars or lower"
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*stars?\b", text)
    if not m:
        return {"min_star_rating": None, "max_star_rating": None}

    val = float(m.group(1))
    before = text[:m.start()]
    after = text[m.end():]

    # Strict max: "below/under/less than" => < val (exclude val)
    strict_max_words_before = ["below", "under", "less than", "lower than"]
    # Inclusive max: "at most/up to/no more than" OR "or lower/or less/and below"
    incl_max_words_before = ["at most", "maximum", "max ", "no more than", "up to"]
    incl_max_words_after = ["or lower", "or less", "and below", "or under"]

    # Strict min: "above/over/greater than/more than" => > val (exclude val)
    strict_min_words_before = ["above", "over", "greater than", "more than", "higher than"]
    # Inclusive min: "at least/minimum/no less than/or higher/or more/and above"
    incl_min_words_before = ["at least", "minimum", "min ", "no less than"]
    incl_min_words_after = ["or higher", "or more", "and above", "+"]

    # Equality: "exactly 4 stars" / "only 4 stars"
    eq_words_before = ["exactly", "only", "just"]

    # Decide
    if any(w in before for w in eq_words_before):
        min_star = val
        max_star = val
    elif any(w in before for w in strict_max_words_before):
        # strictly less than val, approximate using a tiny epsilon
        max_star = val - 0.001
    elif any(w in before for w in incl_max_words_before) or any(w in after for w in incl_max_words_after):
        max_star = val
    elif any(w in before for w in strict_min_words_before):
        # strictly greater than val
        min_star = val + 0.001
    elif any(w in before for w in incl_min_words_before) or any(w in after for w in incl_min_words_after):
        min_star = val
    else:
        # Default meaning of "4 star hotel" usually equals 4 (not >=4)
        # If you prefer it to mean ">=4", change this to min_star = val.
        min_star = val
        max_star = val

    return {"min_star_rating": min_star, "max_star_rating": max_star}




def extract_review_score(user_text: str):
    """
    Extract overall review score constraints.
    Examples:
      - "rating above 8"
      - "score at least 9"
      - "8+ rating"
    We assume scores are on a 0-10 scale.
    """
    text = user_text.lower()
    candidates: List[float] = []

    # Patterns like "8+ rating", "rating 8 or higher"
    pattern1 = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:/10)?\s*(?:rating|score)", text)
    for val in pattern1:
        num = _to_float(val)
        if num is not None:
            candidates.append(num)

    # Phrases like "at least 9", "above 8"
    pattern2 = re.findall(
        r"(?:at least|above|over|greater than|>=)\s+(\d+(?:\.\d+)?)",
        text,
    )
    for val in pattern2:
        num = _to_float(val)
        if num is not None:
            candidates.append(num)

    if not candidates:
        return None

    # Use the largest constraint as min_review_score
    return max(candidates)

def extract_traveller_types_ner(user_text: str):
    doc = nlp(user_text)
    found = set()

    for ent in doc.ents:
        if ent.label_ == "TRAVELLER_TYPE":
            t = ent.text.lower()
            # Normalize to your 4 types
            if "business" in t or "work" in t:
                found.add("business")
            elif "family" in t or "kids" in t or "children" in t:
                found.add("family")
            elif "couple" in t or "wife" in t or "husband" in t or "partner" in t or "spouse" in t:
                found.add("couple")
            elif "solo" in t or "alone" in t or "myself" in t:
                found.add("solo")

    return list(found)

def extract_preferred_aspects(user_text: str):
    """
    Map user phrases to KG review *score* fields:
      - score_cleanliness
      - score_comfort
      - score_facilities
      - score_location
      - score_staff
      - score_value_for_money

    We output the KG property names directly, so Cypher can do things like:
      ORDER BY avg(r[aspect]) DESC
    or use them in WHERE filters.
    """
    text = user_text.lower()
    aspects = set()

    aspect_keywords = {
        "score_cleanliness": ["clean", "cleanliness", "hygiene", "hygienic"],
        "score_comfort": ["comfort", "comfortable", "comfy", "bed", "mattress"],
        "score_facilities": ["facilities", "facility", "amenities", "pool", "gym", "spa"],
        "score_location": ["location", "central", "city center", "downtown", "near metro"],
        "score_staff": ["staff", "service", "reception", "host", "hosts"],
        "score_value_for_money": ["value", "value for money", "cheap", "budget", "affordable"],
    }

    for aspect_field, words in aspect_keywords.items():
        for w in words:
            if w in text:
                aspects.add(aspect_field)
                break

    return list(aspects)



def parse_user_query(user_text):

    data = _llm_extract(user_text)
    intent = data.get("intent") or "hotel_search"

    ner_cities, ner_countries = ner_locations(user_text)
    cities = [c for c in ner_cities if c in CITY_LIST]
    countries = [c for c in ner_countries if c in COUNTRY_LIST]

    hotels = ner_hotels(user_text)

    traveller_types = extract_traveller_types_ner(user_text)

    star_constraints = extract_star_rating(user_text)
    min_star_rating = star_constraints["min_star_rating"]
    max_star_rating = star_constraints["max_star_rating"]

    min_review_score = extract_review_score(user_text)


    preferred_aspects = extract_preferred_aspects(user_text)


    return {
        "intent": intent,
        "cities": cities,
        "countries": countries,
        "hotels": hotels,
        "traveller_types": traveller_types,
        "preferred_aspects": preferred_aspects,
        "min_star_rating": min_star_rating,
        "max_star_rating": max_star_rating,
        "min_review_score": min_review_score,
    }

