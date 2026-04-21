import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an AI surveillance query parser for PhantomEye system.
User gives natural language queries about tracked people in a surveillance scene.
You extract structured filters from the query.

Return ONLY valid JSON with these fields:
{
  "emotion": null or one of ["angry","happy","sad","fear","surprise","disgust","neutral"],
  "gender": null or "Man" or "Woman",
  "min_age": null or integer,
  "max_age": null or integer,
  "min_dwell_seconds": null or integer,
  "loitering": null or true or false,
  "summary": "one line human readable summary of the filter"
}

Examples:
"show me all angry men" -> {"emotion":"angry","gender":"Man","min_age":null,"max_age":null,"min_dwell_seconds":null,"loitering":null,"summary":"Angry men"}
"who was loitering?" -> {"emotion":null,"gender":null,"min_age":null,"max_age":null,"min_dwell_seconds":null,"loitering":true,"summary":"People detected loitering"}
"young women under 30" -> {"emotion":null,"gender":"Woman","min_age":null,"max_age":30,"min_dwell_seconds":null,"loitering":null,"summary":"Women under 30 years old"}
"people who stayed more than 2 minutes" -> {"emotion":null,"gender":null,"min_age":null,"max_age":null,"min_dwell_seconds":120,"loitering":null,"summary":"People with dwell time over 2 minutes"}
"""


def parse_nl_query(query: str) -> dict:
    """
    Convert natural language query to structured filter dict via Groq LLM.
    Returns parsed filter or error dict.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return {"success": True, "filters": parsed}
    except Exception as e:
        return {"success": False, "error": str(e), "filters": {}}


def apply_filters(records: list[dict], filters: dict) -> list[dict]:
    """
    Apply parsed filters to a list of person records.
    Each record: {id, emotion, gender, age, dwell_seconds, loitering}
    Returns filtered list.
    """
    results = records
    if filters.get("emotion"):
        results = [r for r in results if r.get("emotion", "").lower() == filters["emotion"].lower()]
    if filters.get("gender"):
        results = [r for r in results if r.get("gender", "").lower() == filters["gender"].lower()]
    if filters.get("min_age") is not None:
        results = [r for r in results if r.get("age", 0) >= filters["min_age"]]
    if filters.get("max_age") is not None:
        results = [r for r in results if r.get("age", 0) <= filters["max_age"]]
    if filters.get("min_dwell_seconds") is not None:
        results = [r for r in results if r.get("dwell_seconds", 0) >= filters["min_dwell_seconds"]]
    if filters.get("loitering") is not None:
        results = [r for r in results if r.get("loitering", False) == filters["loitering"]]
    return results