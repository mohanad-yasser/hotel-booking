from typing import Dict, List, Any, Tuple
from collections import defaultdict
import time
import requests
import csv
import json
import statistics

from config import HF_API_TOKEN

HF_MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-V3.2",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}


def main_llm_call(
    cypher_answer: str,
    embeddings_model1: str,
    embeddings_model2: str,
    llm_model: str,
    user_query: str,
) -> str:
    """
    Main entrypoint used by your system. Returns ONLY the assistant answer string.
    Uses HuggingFace Inference API (no localhost).
    """
    answer, _metrics = main_llm_call_with_metrics(
        cypher_answer=cypher_answer,
        embeddings_model1=embeddings_model1,
        embeddings_model2=embeddings_model2,
        llm_model=llm_model,
        user_query=user_query,
        temperature=0.2,
    )
    return answer


def main_llm_call_with_metrics(
    cypher_answer: str,
    embeddings_model1: str,
    embeddings_model2: str,
    llm_model: str,
    user_query: str,
    temperature: float = 0.2,
) -> Tuple[str, Dict[str, Any]]:
    """
    Same as main_llm_call but returns (answer, metrics).
    Needed for Milestone d quantitative evaluation.
    """
    # Parse baseline (Cypher) outputs
    cypher_hotels = _parse_cypher(cypher_answer)
    cypher_reviews = _parse_cypher_reviews(cypher_answer)
    visa_req = _parse_visa_requirement_single(cypher_answer)
    visa_free = _parse_visa_free_destinations(cypher_answer)

    # ✅ NEW: Parse "popular hotels for nationality" output (baseline-only intent)
    popular_hotels = _parse_popular_hotels_for_nationality(cypher_answer)

    # Parse embeddings
    emb_hotels_m1 = _parse_embeddings(embeddings_model1)
    emb_hotels_m2 = _parse_embeddings(embeddings_model2)

    for h in emb_hotels_m1:
        h["emb_model"] = "model1"
    for h in emb_hotels_m2:
        h["emb_model"] = "model2"

    emb_hotels = emb_hotels_m1 + emb_hotels_m2

    # Scenario selection. VISA overrides everything and forces baseline-only.
    if visa_req is not None:
        scenario, details = "VISA_REQUIREMENT", "Visa requirement detected in Cypher output."
    elif visa_free:
        scenario, details = "VISA_FREE_LIST", "Visa-free destinations detected in Cypher output."
    elif popular_hotels:
        # ✅ NEW: Baseline-only scenario
        scenario, details = "POPULAR_HOTELS_FOR_NATIONALITY", "Popularity-by-nationality results detected in Cypher output."
    else:
        scenario, details = _detect_scenario(cypher_hotels, emb_hotels, cypher_reviews)

    combined_hotels = _combine_hotels(cypher_hotels, emb_hotels)

    # Build prompt context
    context_lines: List[str] = []

    # 0) Visa context first (very explicit)
    if visa_req is not None:
        context_lines.append("Visa requirements:")
        context_lines.append(f"- Visa type required: {visa_req.get('visa_type', 'Unknown')}")
        context_lines.append("")

    if visa_free:
        context_lines.append("Visa-free destinations:")
        for item in visa_free:
            c = item.get("country", "Unknown")
            vt = item.get("visa_type", None)
            if vt:
                context_lines.append(f"- {c} ({vt})")
            else:
                context_lines.append(f"- {c}")
        context_lines.append("")

    # ✅ NEW: Popular hotels for nationality block
    if popular_hotels:
        context_lines.append("Popular hotels for nationality:")
        for h in popular_hotels:
            line = (
                f"- {h.get('name', 'Unknown')}"
                f" | City: {h.get('city', 'Unknown')}"
                f" | Country: {h.get('country', 'Unknown')}"
                f" | Stars: {h.get('stars', 'Unknown')}"
                f" | Avg Review: {h.get('avg_review', 'Unknown')}"
            )
            # optional
            if h.get("cleanliness") is not None:
                line += f" | Cleanliness: {h.get('cleanliness')}"
            if h.get("comfort") is not None:
                line += f" | Comfort: {h.get('comfort')}"
            if h.get("facilities") is not None:
                line += f" | Facilities: {h.get('facilities')}"
            if h.get("stay_count") is not None:
                line += f" | Stay Count: {h.get('stay_count')}"
            context_lines.append(line)
        context_lines.append("")

    # 1) Reviews
    if cypher_reviews:
        context_lines.append("Reviews (latest):")
        for r in cypher_reviews:
            context_lines.append(
                f"- Hotel: {r.get('hotel','Unknown')}"
                f" | Traveller type: {r.get('traveller_type','Unknown')}"
                f" | Score: {r.get('score','Unknown')}"
                f" | Review: {r.get('review','')}"
            )
        context_lines.append("")

    # 2) Hotels
    if combined_hotels:
        context_lines.append("Hotels (merged from retrieval):")
        for h in combined_hotels.values():
            line = (
                f"- {h.get('name', 'Unknown')}"
                f" | City: {h.get('city', 'Unknown')}"
                f" | Country: {h.get('country', 'Unknown')}"
                f" | Stars: {h.get('stars', 'Unknown')}"
                f" | Avg Review: {h.get('avg_review', 'Unknown')}"
                f" | Cleanliness: {h.get('cleanliness', 'Unknown')}"
                f" | Comfort: {h.get('comfort', 'Unknown')}"
                f" | Facilities: {h.get('facilities', 'Unknown')}"
                f" | Similarity: {h.get('sim', 'Unknown')}"
            )
            context_lines.append(line)
    else:
        # If no hotels and no reviews and no visa and no popular block, fall back to raw KG text
        if not cypher_reviews and visa_req is None and not visa_free and not popular_hotels:
            raw_kg = (cypher_answer or "").strip()
            raw_emb1 = (embeddings_model1 or "").strip()
            raw_emb2 = (embeddings_model2 or "").strip()

            if raw_kg or raw_emb1 or raw_emb2:
                context_lines.append("Raw KG context (non-hotel structured):")
                if raw_kg:
                    context_lines.append("Cypher result:")
                    context_lines.append(raw_kg)

                if raw_emb1:
                    context_lines.append("")
                    context_lines.append("Embeddings result (Model 1):")
                    context_lines.append(raw_emb1)

                if raw_emb2:
                    context_lines.append("")
                    context_lines.append("Embeddings result (Model 2):")
                    context_lines.append(raw_emb2)
            else:
                context_lines.append("KG context: (none retrieved)")

    # Retrieval summary
    context_lines.append("")
    context_lines.append("Retrieval summary:")
    context_lines.append(f"- Scenario: {scenario}")
    if details:
        context_lines.append(f"- Details: {details}")

    context = "\n".join(context_lines)

    # Persona (keep your strict “no meta talk” rules)
    persona = (
        "You are a helpful travel assistant. "
        "Use ONLY the information in the Context. "
        "You can explain the info, but don't add any extra information. "
        "NEVER mention the words 'context', 'provided context', 'knowledge graph', 'KG', 'retrieval', 'Cypher', or 'embeddings'. "
        "NEVER explain your reasoning process. "
        "If information is missing, say: 'I don’t have that information.' "
        "Answer directly as if you already know the facts."
    )

    task = _build_task_for_scenario(scenario)

    messages = [
        {"role": "system", "content": persona},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"User question:\n{user_query}\n\n"
                f"Task:\n{task}"
            ),
        },
    ]

    model_id = (llm_model or "").strip().lower()
    if model_id not in HF_MODELS:
        raise ValueError(f"Unknown model: {llm_model}. Use one of: {list(HF_MODELS.keys())}")

    answer, metrics = _hf_chat(
        model_name=HF_MODELS[model_id],
        messages=messages,
        temperature=temperature,
    )

    metrics = dict(metrics)
    metrics["model_key"] = model_id
    metrics["model_name"] = HF_MODELS[model_id]
    metrics["scenario"] = scenario

    return answer, metrics


def _build_task_for_scenario(scenario: str) -> str:
    if scenario == "VISA_REQUIREMENT":
        return (
            "This is a visa question. Use ONLY the visa information. "
            "Ignore anything else. Answer with the visa type exactly."
        )

    if scenario == "VISA_FREE_LIST":
        return (
            "This is a visa-free destinations question. Use ONLY the list. "
            "Return the destinations as bullet points. Do not invent countries."
        )

  
    if scenario == "POPULAR_HOTELS_FOR_NATIONALITY":
        return (
            "The user asked for hotels popular among travellers from a nationality. "
            "These hotels are already filtered as popular among travellers from the requested nationality",
            "This is the information you have on popular hotels in a country for travelers from a nationality."
            "Return the hotels as bullet points with their city, country, stars, and avg review if present. "
            "Do not invent hotels or claims."
        )

    if scenario == "REVIEWS_ONLY":
        return (
            "Return the reviews as a bullet list. "
            "Do not invent reviews. Do not paraphrase the review text."
        )

    if scenario == "NO_KG_ANSWER":
        return (
            "There is no relevant answer available. "
            "Say you don’t have that information. Ask ONE short follow-up question."
        )

    if scenario == "CYPHER_ONLY":
        return (
            "Answer using ONLY baseline hotel results. "
            "Do not use similarity as evidence."
        )

    if scenario == "EMBEDDINGS_ONLY":
        return (
            "Answer using ONLY embeddings results. "
            "Say they are similarity-based candidates and may not exactly match constraints."
        )

    if scenario == "CONTRADICTION":
        return (
            "Prefer baseline hotel facts. "
            "Embeddings can be mentioned only as possible alternatives."
        )

    if scenario == "CYPHER_MORE_RELIABLE":
        return (
            "Base your answer on baseline hotels first, embeddings second. "
            "Do not let embeddings override constraints."
        )

    return "Answer the user question using only the provided information. Do not invent facts."


def _detect_scenario(
    cypher_hotels: List[Dict[str, Any]],
    emb_hotels: List[Dict[str, Any]],
    cypher_reviews: List[Dict[str, Any]],
) -> Tuple[str, str]:
    if cypher_reviews and not cypher_hotels and not emb_hotels:
        return "REVIEWS_ONLY", "Cypher returned latest reviews, no hotel candidates."

    if not cypher_hotels and not emb_hotels and not cypher_reviews:
        return "NO_KG_ANSWER", "No results from Cypher or embeddings."

    if cypher_hotels and not emb_hotels:
        return "CYPHER_ONLY", "Only Cypher returned results."

    if emb_hotels and not cypher_hotels:
        return "EMBEDDINGS_ONLY", "Only embeddings returned results."

    cy_names = {h["name"].strip().lower() for h in cypher_hotels if h.get("name")}
    em_names = {h["name"].strip().lower() for h in emb_hotels if h.get("name")}
    overlap = cy_names.intersection(em_names)

    sims = [h.get("sim") for h in emb_hotels if isinstance(h.get("sim"), (int, float))]
    top_sim = max(sims) if sims else None

    if not overlap and top_sim is not None and top_sim >= 0.70:
        return "CONTRADICTION", f"No overlap. Top embedding similarity={top_sim:.2f}."

    if top_sim is None or top_sim < 0.55:
        return "CYPHER_MORE_RELIABLE", f"Embeddings seem weak. Top similarity={top_sim}."

    return "BOTH_OK", f"Overlap size={len(overlap)}. Top similarity={top_sim}."


def _parse_cypher(cypher_answer: str) -> List[Dict[str, Any]]:
    if not cypher_answer:
        return []

    hotels: List[Dict[str, Any]] = []

    for raw_line in cypher_answer.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if line.lower().startswith("hotels:"):
            continue

        if line.startswith("- "):
            line = line[2:].strip()

        if " | " not in line:
            continue

        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) < 2:
            continue

        first = parts[0].strip().lower()
        if first.startswith("hotel:"):
            continue
        if first.startswith("visa type required:"):
            continue
        if first.startswith("stay count:"):
            continue

        name = parts[0]
        h: Dict[str, Any] = {"name": name}

        for p in parts[1:]:
            if ": " not in p:
                continue
            k, v = p.split(": ", 1)
            key = k.strip().lower().replace(" ", "_")
            val_raw = v.strip()

            val: Any = val_raw
            try:
                val = float(val_raw)
            except Exception:
                val = val_raw

            h[key] = val

        h.setdefault("city", "Unknown")
        h.setdefault("country", "Unknown")
        h.setdefault("stars", None)
        h.setdefault("avg_review", None)

        hotels.append(h)

    return hotels


def _parse_popular_hotels_for_nationality(cypher_answer: str) -> List[Dict[str, Any]]:
    """
    Expects something like:

    Popular hotels for nationality:
    - HotelName | City: X | Country: Y | Stars: 5 | Avg Review: 9.1 | Stay Count: 12 | Cleanliness: 9.0 ...

    If your header is different, change the header check string below.
    """
    if not cypher_answer:
        return []

    lines = cypher_answer.splitlines()
    header_idx = None
    for i, l in enumerate(lines):
        if (l or "").strip().lower().startswith("popular hotels for nationality"):
            header_idx = i
            break

    if header_idx is None:
        return []

    out: List[Dict[str, Any]] = []
    for raw_line in lines[header_idx + 1 :]:
        line = (raw_line or "").strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()

        if " | " not in line:
            continue

        parts = [p.strip() for p in line.split(" | ") if p.strip()]
        if not parts:
            continue

        h: Dict[str, Any] = {"name": parts[0]}

        for p in parts[1:]:
            if ": " not in p:
                continue
            k, v = p.split(": ", 1)
            key = k.strip().lower().replace(" ", "_")
            val_raw = v.strip()

            val: Any = val_raw
            try:
                # stay count can be int
                if key in ("stay_count", "count", "traveller_count"):
                    val = int(float(val_raw))
                else:
                    val = float(val_raw)
            except Exception:
                val = val_raw

            # normalize keys to match hotel printing
            if key == "avg_review":
                h["avg_review"] = val
            elif key == "stars":
                h["stars"] = val
            elif key == "cleanliness":
                h["cleanliness"] = val
            elif key == "comfort":
                h["comfort"] = val
            elif key == "facilities":
                h["facilities"] = val
            elif key == "city":
                h["city"] = val
            elif key == "country":
                h["country"] = val
            elif key == "stay_count":
                h["stay_count"] = val
            else:
                h[key] = val

        out.append(h)

    return out


def _parse_cypher_reviews(cypher_answer: str) -> List[Dict[str, Any]]:
    if not cypher_answer:
        return []

    lines = cypher_answer.splitlines()
    looks_like_reviews = any("reviews" in (l or "").strip().lower() for l in lines)
    if not looks_like_reviews:
        return []

    reviews: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}

    def flush_current():
        nonlocal current
        if current and current.get("hotel") and current.get("review") is not None:
            current["review"] = current["review"].strip()
            reviews.append(current)
        current = {}

    for raw in lines:
        line = (raw or "").rstrip()
        if not line.strip():
            if current.get("review") is not None:
                current["review"] += "\n"
            continue

        if line.strip().lower().startswith("reviews"):
            continue

        if line.strip().startswith("- "):
            flush_current()
            item = line.strip()[2:].strip()
            parts = [p.strip() for p in item.split(" | ") if p.strip()]

            data: Dict[str, Any] = {"hotel": None, "traveller_type": None, "score": None, "review": ""}

            for p in parts:
                if ": " not in p:
                    continue
                k, v = p.split(": ", 1)
                k_norm = k.strip().lower()

                if k_norm == "hotel":
                    data["hotel"] = v.strip()
                elif k_norm in ("traveller type", "traveler type"):
                    data["traveller_type"] = v.strip()
                elif k_norm == "score":
                    try:
                        data["score"] = float(v.strip())
                    except Exception:
                        data["score"] = v.strip()
                elif k_norm == "review":
                    data["review"] = v.rstrip()

            current = data
        else:
            if current.get("review") is not None:
                if current["review"] and not current["review"].endswith("\n"):
                    current["review"] += "\n"
                current["review"] += line.strip()

    flush_current()
    return reviews


def _parse_embeddings(embeddings_answer: str) -> List[Dict[str, Any]]:
    if not embeddings_answer:
        return []

    hotels: List[Dict[str, Any]] = []
    for raw_line in embeddings_answer.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("- "):
            line = line[2:].strip()

        if " | " not in line:
            continue

        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) < 3:
            continue

        name = parts[0]
        location = parts[1]
        sim_part = parts[2]

        sim = None
        if "=" in sim_part:
            try:
                sim = float(sim_part.split("=", 1)[1].strip())
            except Exception:
                sim = None

        if ", " in location:
            city, country = location.split(", ", 1)
        else:
            city, country = location, "Unknown"

        hotels.append({"name": name, "city": city, "country": country, "sim": sim})

    return hotels


def _parse_visa_requirement_single(cypher_answer: str) -> Dict[str, Any] | None:
    if not cypher_answer:
        return None

    text = cypher_answer.strip()
    if "visa requirements" not in text.lower():
        return None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("visa requirements"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line.lower().startswith("visa type required:"):
            visa_type = line.split(":", 1)[1].strip()
            return {"visa_type": visa_type}

    return None


def _parse_visa_free_destinations(cypher_answer: str) -> List[Dict[str, Any]]:
    if not cypher_answer:
        return []

    text = cypher_answer.strip()
    if "visa-free destinations" not in text.lower():
        return []

    out: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("visa-free destinations"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()

        if "(" in line and line.endswith(")"):
            country = line[: line.rfind("(")].strip()
            visa_type = line[line.rfind("(") + 1 : -1].strip()
            if country:
                out.append({"country": country, "visa_type": visa_type})
        else:
            out.append({"country": line, "visa_type": None})

    return out


def _combine_hotels(
    cypher_hotels: List[Dict[str, Any]],
    emb_hotels: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    combined = defaultdict(dict)

    def key_of(name: str) -> str:
        return name.strip().lower()

    for h in cypher_hotels:
        if not h.get("name"):
            continue
        combined[key_of(h["name"])].update(h)

    for h in emb_hotels:
        if not h.get("name"):
            continue
        combined[key_of(h["name"])].update(h)

    out: Dict[str, Dict[str, Any]] = {}
    for k, v in combined.items():
        display = v.get("name") or k
        out[display] = v
    return out


def _hf_chat(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_new_tokens: int = 512,
) -> Tuple[str, Dict[str, Any]]:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not set. Put it in config.py as HF_API_TOKEN='...'")

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    resp = requests.post(
        url,
        headers=headers,
        json={
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
        },
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"HF request failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = str(data)

    metrics = {
        "latency_ms": latency_ms,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cost_usd": 0.0,
    }

    return text.strip(), metrics


def _accuracy_proxy(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0
    a = (answer or "").lower()
    hits = 0
    for kw in expected_keywords:
        if kw.lower() in a:
            hits += 1
    return hits / len(expected_keywords)


def run_llm_benchmark(test_cases: List[Dict[str, Any]]) -> None:
    models = ["deepseek", "llama", "gemma"]
    results: List[Dict[str, Any]] = []
    human_eval: List[Dict[str, Any]] = []

    for tc in test_cases:
        for model in models:
            answer, metrics = main_llm_call_with_metrics(
                cypher_answer=tc.get("cypher_answer", ""),
                embeddings_model1=tc.get("embeddings_answer_model1", ""),
                embeddings_model2=tc.get("embeddings_answer_model2", ""),
                llm_model=model,
                user_query=tc.get("user_query", ""),
                temperature=0.2,
            )

            acc = _accuracy_proxy(answer, tc.get("expected_keywords", []))

            results.append(
                {
                    "test_id": tc.get("id", ""),
                    "model": model,
                    "scenario": metrics.get("scenario"),
                    "latency_ms": metrics.get("latency_ms"),
                    "accuracy_proxy": acc,
                    "cost_usd": metrics.get("cost_usd"),
                    "answer": answer,
                }
            )

            human_eval.append(
                {
                    "test_id": tc.get("id", ""),
                    "model": model,
                    "correctness_1to5": "",
                    "relevance_1to5": "",
                    "naturalness_1to5": "",
                    "groundedness_1to5": "",
                    "notes": "",
                    "user_query": tc.get("user_query", ""),
                    "answer": answer,
                }
            )

    if results:
        with open("llm_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    if human_eval:
        with open("llm_human_eval.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(human_eval[0].keys()))
            w.writeheader()
            w.writerows(human_eval)

    summary: Dict[str, Any] = {}
    for m in models:
        rows = [r for r in results if r["model"] == m]
        if not rows:
            continue

        latencies = [r["latency_ms"] for r in rows if isinstance(r["latency_ms"], (int, float))]
        accuracies = [r["accuracy_proxy"] for r in rows if isinstance(r["accuracy_proxy"], (int, float))]
        costs = [r["cost_usd"] for r in rows if isinstance(r["cost_usd"], (int, float))]

        summary[m] = {
            "avg_latency_ms": statistics.mean(latencies) if latencies else None,
            "avg_accuracy_proxy": statistics.mean(accuracies) if accuracies else None,
            "total_cost_usd": sum(costs) if costs else 0.0,
        }

    with open("llm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Benchmark completed. Outputs: llm_results.csv, llm_human_eval.csv, llm_summary.json")
