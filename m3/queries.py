# queries.py

from typing import Dict, Any, List


# --------------------------------------------------------------------
# Template 1: Basic hotel search in a city (with optional min star / min review)
# --------------------------------------------------------------------
def template_1_hotels_in_city(
    city: str,
    min_star: float | None = None,
    max_star: float | None = None,
    min_review: float | None = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Template 1: Find hotels in a given city with optional rating filters.

    User example:
        "Show me hotels in Cairo with at least 4 stars and review score above 8"

    Args:
        city: City name (e.g., "Cairo")
        min_star: Minimum star rating (optional)
        min_review: Minimum average review score (optional)
        limit: Max number of hotels

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)
    WHERE toLower(city.name) = toLower($city)
      AND ($min_star IS NULL OR h.star_rating >= $min_star)
      AND ($max_star IS NULL OR h.star_rating <= $max_star)
      AND ($min_review IS NULL OR h.average_reviews_score >= $min_review)
    RETURN   
        h AS h,
        city AS city,
        h.star_rating AS star_rating,
        h.average_reviews_score AS average_reviews_score,
        h.cleanliness_base AS cleanliness,
        h.comfort_base AS comfort,
        h.facilities_base AS facilities
    ORDER BY h.average_reviews_score DESC, h.star_rating DESC
    LIMIT $limit
    """
    params = {
        "city": city,
        "min_star": min_star,
        "min_review": min_review,
        "max_star": max_star, 
        "limit": limit,
    }
    return {"query": query, "params": params}


# --------------------------------------------------------------------
# Template 2: Hotel search in a country (optionally filtered by city list)
# --------------------------------------------------------------------
def template_2_hotels_in_country(
    country: str,
    cities: List[str] | None = None,
    min_star: float | None = None,
    max_star: float | None = None,
    min_review: float | None = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Template 2: Find hotels in a given country, optionally restricted to some cities.

    User example:
        "Find good hotels in Italy, especially in Rome or Milan, with rating above 8"

    Args:
        country: Country name (e.g., "Italy")
        cities: Optional list of city names to restrict search
        min_star: Minimum star rating (optional)
        min_review: Minimum average review score (optional)
        limit: Max number of hotels

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
    WHERE toLower(country.name) = toLower($country)
      AND (size($cities) = 0 OR toLower(city.name) IN $cities)
      AND ($min_star IS NULL OR h.star_rating >= $min_star)
      AND ($max_star IS NULL OR h.star_rating <= $max_star)
      AND ($min_review IS NULL OR h.average_reviews_score >= $min_review)
    RETURN   
        h AS h,
        city AS city,
        country AS country,
        h.star_rating AS star_rating,
        h.average_reviews_score AS average_reviews_score,
        h.cleanliness_base AS cleanliness,
        h.comfort_base AS comfort,
        h.facilities_base AS facilities
    ORDER BY h.average_reviews_score DESC, h.star_rating DESC
    LIMIT $limit
    """
    params = {
        "country": country,
        "cities": [c.lower() for c in (cities or [])],
        "min_star": min_star,
        "max_star": max_star, 
        "min_review": min_review,
        "limit": limit,
    }
    return {"query": query, "params": params}


# --------------------------------------------------------------------
# Template 4: Latest reviews for a specific hotel
# --------------------------------------------------------------------
def template_4_latest_reviews_for_hotel(
    hotel_name: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Template 4: Get the most recent reviews for a given hotel.

    User example:
        "Show me the latest reviews for The Golden Oasis"

    Args:
        hotel_name: Exact or case-insensitive hotel name
        limit: Max number of reviews

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
    MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)
    OPTIONAL MATCH (t:Traveller)-[:WROTE]->(r)
    RETURN h, r, t
    ORDER BY r.date DESC
    LIMIT $limit
    """
    params = {
        "hotel_name": hotel_name,
        "limit": limit,
    }
    return {"query": query, "params": params}


# --------------------------------------------------------------------
# Template 5: Most popular hotels for travellers from a given nationality
# --------------------------------------------------------------------
def template_5_popular_hotels_for_nationality(
    nationality_country: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Template 5: Get the most popular hotels among travellers from a specific country.

    User example:
        "What are the most popular hotels in Dubai for travellers from Egypt?"

    Args:
        nationality_country: Country name of traveller origin
        limit: Max number of hotels

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
    MATCH (t:Traveller)-[:FROM_COUNTRY]->(orig:Country)
    MATCH (t)-[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(dest:Country)
    WHERE toLower(orig.name) = toLower($nationality_country)
    WITH h, city, dest, count(DISTINCT t) AS stay_count,
         avg(h.average_reviews_score) AS avg_score
    RETURN   
        h AS h,
        city AS city,
        country AS country,
        h.star_rating AS star_rating,
        h.average_reviews_score AS average_reviews_score,
        h.cleanliness_base AS cleanliness,
        h.comfort_base AS comfort,
        h.facilities_base AS facilities
    ORDER BY stay_count DESC, avg_score DESC
    LIMIT $limit
    """
    params = {
        "nationality_country": nationality_country,
        "limit": limit,
    }
    return {"query": query, "params": params}


# --------------------------------------------------------------------
# Template 6: Visa requirement from country A to country B
# --------------------------------------------------------------------
def template_6_visa_requirements(
    from_country: str,
    to_country: str,
) -> Dict[str, Any]:
    """
    Template 6: Check visa requirements between two countries.

    User example:
        "Do I need a visa to travel from Egypt to Turkey?"

    Args:
        from_country: Origin country
        to_country: Destination country

    Returns:
        Dict with 'query' and 'params'
    """
    # Assuming relation (:Country {name: from})-[:NEEDS_VISA {visa_type}]->(:Country {name: to})
    query = """
    MATCH (from:Country)-[rel:NEEDS_VISA]->(to:Country)
    WHERE toLower(from.name) = toLower($from_country)
      AND toLower(to.name) = toLower($to_country)
    RETURN from, to, rel.visa_type AS visa_type
    """
    params = {
        "from_country": from_country,
        "to_country": to_country,
    }
    return {"query": query, "params": params}


# --------------------------------------------------------------------
# Template 9: Visa-free destinations from a given country
# --------------------------------------------------------------------
def template_9_visa_free_destinations(
    from_country: str,
) -> Dict[str, Any]:
    """
    Template 9: List all destinations that are visa-free (or special visa type)
    for travellers from a given country.

    User example:
        "Which countries can I travel to from Egypt without a visa?"

    NOTE:
        You can later restrict by specific visa_type if your schema distinguishes
        'no_visa', 'on_arrival', etc.

    Args:
        from_country: Origin country name

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
    MATCH (from:Country)-[rel:NEEDS_VISA]->(to:Country)
    WHERE toLower(from.name) = toLower($from_country)
      AND rel.visa_type = 'NO_VISA'
    RETURN from, to, rel.visa_type AS visa_type
    ORDER BY to.name
    """
    params = {"from_country": from_country}
    return {"query": query, "params": params}
# --------------------------------------------------------------------
# Template 3: Hotels based on rating criteria (star_rating)
# --------------------------------------------------------------------
def template_3_hotels_by_rating_option(
    rating_option: str,
    city: str | None = None,
    country: str | None = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Template 3: Get hotels based on star rating criteria.

    Rating options:
        "4+"   → star_rating >= 4
        "-4"   → star_rating < 4
        "4"    → star_rating = 4
        "3-4"  → 3 <= star_rating <= 4

    User examples:
        "Show me 4 star hotels in Paris"
        "Find hotels below 4 stars in Istanbul"
        "Show me hotels between 3 and 4 stars in Dubai"

    Args:
        rating_option: One of "4+", "-4", "4", "3-4" (you can extend this mapping)
        city: Optional city filter
        country: Optional country filter
        limit: Max number of hotels

    Returns:
        Dict with 'query' and 'params'
    """
    # Map rating_option to numerical bounds
    min_star = None
    max_star = None

    if rating_option == "4+":
        min_star = 4.0
    elif rating_option == "-4":
        max_star = 4.0
    elif rating_option == "4":
        min_star = 4.0
        max_star = 4.0
    elif rating_option == "3-4":
        min_star = 3.0
        max_star = 4.0

    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
    WHERE ($city IS NULL OR toLower(city.name) = toLower($city))
      AND ($country IS NULL OR toLower(country.name) = toLower($country))
      AND ($min_star IS NULL OR h.star_rating >= $min_star)
      AND ($max_star IS NULL OR h.star_rating <= $max_star)
    RETURN   
        h AS h,
        city AS city,
        country AS country,
        h.star_rating AS star_rating,
        h.average_reviews_score AS average_reviews_score,
        h.cleanliness_base AS cleanliness,
        h.comfort_base AS comfort,
        h.facilities_base AS facilities
    ORDER BY h.star_rating DESC, h.average_reviews_score DESC
    LIMIT $limit
    """
    params = {
        "city": city,
        "country": country,
        "min_star": min_star,
        "max_star": max_star,
        "limit": limit,
    }
    return {"query": query, "params": params}


# --------------------------------------------------------------------
# Template 7: Hotels with highest overall score for traveller type + age group
# --------------------------------------------------------------------
def template_7_hotels_for_traveller_type_and_age(
    traveller_type: str | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    city: str | None = None,
    country: str | None = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Template 7: Get hotels that perform best for a given traveller type and age group.

    User examples:
        "Best hotels for families with kids in Dubai"
        "Top hotels for solo travellers aged 25-35 in Paris"
        "What hotels do business travellers aged 30+ like in London?"

    Args:
        traveller_type: 'family', 'business', 'solo', 'couple', 'group', etc. (optional)
        min_age: Minimum traveller age (optional)
        max_age: Maximum traveller age (optional)
        city: Optional city filter
        country: Optional country filter
        limit: Max number of hotels

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
    MATCH (h)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
    WHERE ($traveller_type IS NULL OR toLower(t.type) = toLower($traveller_type))
      AND ($min_age IS NULL OR t.age >= $min_age)
      AND ($max_age IS NULL OR t.age <= $max_age)
      AND ($city IS NULL OR toLower(city.name) = toLower($city))
      AND ($country IS NULL OR toLower(country.name) = toLower($country))
    WITH h, city, country,
         avg(r.score_overall) AS avg_overall,
         count(DISTINCT t) AS traveller_count
    RETURN   
        h AS h,
        city AS city,
        country AS country,
        h.star_rating AS star_rating,
        h.average_reviews_score AS average_reviews_score,
        h.cleanliness_base AS cleanliness,
        h.comfort_base AS comfort,
        h.facilities_base AS facilities
    ORDER BY avg_overall DESC, traveller_count DESC
    LIMIT $limit
    """
    params = {
        "traveller_type": traveller_type,
        "min_age": min_age,
        "max_age": max_age,
        "city": city,
        "country": country,
        "limit": limit,
    }
    return {"query": query, "params": params}


# --------------------------------------------------------------------
# Template 8: Best hotels by specific score criteria (dynamic aspects)
# --------------------------------------------------------------------
def template_8_hotels_by_aspect_scores(
    aspects: List[str],
    city: str | None = None,
    country: str | None = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Template 8: Get best hotels based on specific aspect scores,
    BUT using Hotel base values (cleanliness_base, comfort_base, facilities_base).

    Supported mappings:
      - 'score_cleanliness' -> h.cleanliness_base
      - 'score_comfort'     -> h.comfort_base
      - 'score_facilities'  -> h.facilities_base

    For aspects that do not exist as Hotel base properties in your KG
    (score_location, score_staff, score_value_for_money), we return NULL
    so they get ignored in scoring.

    The final_score is the average of the selected (non-null) base values.
    If no usable aspects were provided, it falls back to h.average_reviews_score.

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
    WHERE ($city IS NULL OR toLower(city.name) = toLower($city))
      AND ($country IS NULL OR toLower(country.name) = toLower($country))

    WITH h, city, country,
         [
            CASE WHEN 'score_cleanliness' IN $aspects THEN h.cleanliness_base ELSE NULL END,
            CASE WHEN 'score_comfort'     IN $aspects THEN h.comfort_base     ELSE NULL END,
            CASE WHEN 'score_facilities'  IN $aspects THEN h.facilities_base ELSE NULL END,

            
            CASE WHEN 'score_location'        IN $aspects THEN NULL ELSE NULL END,
            CASE WHEN 'score_staff'           IN $aspects THEN NULL ELSE NULL END,
            CASE WHEN 'score_value_for_money' IN $aspects THEN NULL ELSE NULL END
         ] AS aspect_values

    WITH h, city, country,
         [v IN aspect_values WHERE v IS NOT NULL] AS used_values

    WITH h, city, country,
         CASE
            WHEN size(used_values) = 0
                 THEN coalesce(h.average_reviews_score, 0.0)
            ELSE reduce(s = 0.0, v IN used_values | s + v) * 1.0 / size(used_values)
         END AS final_score

    RETURN   
        h AS h,
        city AS city,
        country AS country,
        h.star_rating AS star_rating,
        h.average_reviews_score AS average_reviews_score,
        h.cleanliness_base AS cleanliness,
        h.comfort_base AS comfort,
        h.facilities_base AS facilities
    ORDER BY
        final_score DESC,
        h.average_reviews_score DESC,
        h.star_rating DESC
    LIMIT $limit
    """
    params = {
        "aspects": aspects,
        "city": city,
        "country": country,
        "limit": limit,
    }

    return {"query": query, "params": params}

# --------------------------------------------------------------------
# Template 10: Compare multiple hotels based on ratings and reviews
# --------------------------------------------------------------------
def template_10_compare_hotels(
    hotel_names: List[str],
) -> Dict[str, Any]:
    """
    Template 10: Compare multiple hotels based on ratings and review aspects.

    User examples:
        "Compare The Golden Oasis and The Bosphorus Inn"
        "Compare Marriott Downtown and Hilton City Center in terms of reviews"

    Args:
        hotel_names: List of hotel names to compare (2 or more)

    Returns:
        Dict with 'query' and 'params'
    """
    query = """
MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
WHERE toLower(h.name) IN $hotel_names
OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
WITH h, city, country,
     avg(r.score_overall)          AS avg_overall,
     avg(r.score_cleanliness)      AS avg_cleanliness,
     avg(r.score_comfort)          AS avg_comfort,
     avg(r.score_facilities)       AS avg_facilities,
     avg(r.score_location)         AS avg_location,
     avg(r.score_staff)            AS avg_staff,
     avg(r.score_value_for_money)  AS avg_value_for_money,
     count(r)                      AS review_count
RETURN
  h,
  city,
  country,
  h.star_rating           AS star_rating,
  h.average_reviews_score AS avg_review_from_hotel,

  h.cleanliness_base      AS cleanliness,
  h.comfort_base          AS comfort,
  h.facilities_base       AS facilities,

  avg_overall,
  avg_cleanliness,
  avg_comfort,
  avg_facilities,
  avg_location,
  avg_staff,
  avg_value_for_money,
  review_count

ORDER BY avg_overall DESC, review_count DESC

    """
    params = {
        "hotel_names": [name.lower() for name in hotel_names],
    }
    return {"query": query, "params": params}