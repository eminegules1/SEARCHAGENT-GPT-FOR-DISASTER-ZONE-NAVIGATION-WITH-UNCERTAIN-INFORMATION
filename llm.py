import json
import requests

from config import GRID_N

def get_llama_analysis(user_input: str) -> dict:
    url = "http://localhost:11434/api/generate"

    system_instruction = (
        "You are an AI Search & Rescue Coordinator. Analyze the situational report.\n"
        "Your goal is to convert descriptions into PRECISE coordinates.\n\n"
        "1. HAZARDS (RISKY BUT PASSABLE CELLS):\n"
        "   - 'fire', 'flame' -> risk: 1.0\n"
        "   - 'debris', 'rubble' -> risk: 0.7\n"
        "   - 'water', 'smoke' -> risk: 0.4\n"
        "   - RADIUS RULE (CRITICAL): Default radius is ALWAYS 0 (Single 1x1 Square).\n"
        "     ONLY use radius: 1 if user explicitly says 'huge', 'massive', 'spreading', 'big area'.\n\n"
        "2. BLOCKED CELLS (IMPASSABLE):\n"
        "   - If the report indicates a cell/area is NOT traversable (e.g., 'collapsed road', 'blocked path',\n"
        "     'cannot pass', 'closed street', 'wall of rubble', 'impassable'), output it as BLOCKED.\n"
        "   - Blocked cells must be listed in \"blocked\" as coordinates: [[x,y], [x,y], ...].\n"
        "   - If no blocked areas are mentioned, output an empty list: \"blocked\": [].\n\n"
        "3. SPATIAL MAPPING:\n"
        "   - 'Top Right' -> [MAX, MAX], 'Bottom Left' -> [0, 0], 'Center' -> [MID, MID]\n"
        "   - 'North' -> [MID, MAX], 'South' -> [MID, 0], 'East' -> [MAX, MID], 'West' -> [0, MID]\n\n"
        "4. SURVIVOR:\n"
        "   - Extract location [x, y].\n\n"
        "Return ONLY valid JSON. No markdown, no extra text.\n"
        "Output Format (JSON ONLY):\n"
        "{\n"
        "  \"hazards\": [{\"coord\": [x, y], \"risk\": float, \"radius\": int}],\n"
        "  \"blocked\": [[x, y], [x, y], ...],\n"
        "  \"survivor\": [x, y] or null,\n"
        "  \"direction_bias\": \"north\" or null,\n"
        "  \"summary\": \"Brief explanation\"\n"
        "}"
    )

    # LLM'e grid boyutunu ipucu olarak verelim (çok yardımcı olur)
    maxv = GRID_N - 1
    mid = GRID_N // 2
    system_instruction = system_instruction.replace("MAX", str(maxv)).replace("MID", str(mid))

    prompt_text = f"{system_instruction}\n\nReport: {user_input}\n\nResponse:"

    payload = {
        "model": "llama3.1",
        "prompt": prompt_text,
        "stream": False,
        "format": "json",
        # "options": {"temperature": 0.1, "num_predict": 200},
    }

    try:
        response = requests.post(url, json=payload, timeout=(5, 120))
        response.raise_for_status()

        raw = response.json().get("response", "{}")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            parsed = {}

        parsed.setdefault("hazards", [])
        parsed.setdefault("blocked", [])
        parsed.setdefault("survivor", None)
        parsed.setdefault("direction_bias", None)
        parsed.setdefault("summary", "")

        return parsed

    except Exception as e:
        print(f"Error (LLM): {e}")
        return {
            "hazards": [],
            "blocked": [],
            "survivor": None,
            "direction_bias": None,
            "summary": f"LLM error/timeout: {e}",
        }


# =============================================================================
# 2) RISK BOUNDING + VALIDATION
# =============================================================================
def _in_bounds(x: int, y: int, n: int = GRID_N) -> bool:
    return 0 <= x < n and 0 <= y < n


def _clamp01(v) -> float:
    try:
        v = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def sanitize_llm_data(data: dict, n: int = GRID_N) -> dict:
    """
    Risk Bounding Layer:
    - Drop invalid/out-of-bounds coords
    - Clamp risk to [0, 1]
    - Ensure radius >= 0
    - Parse blocked as a list of valid coords
    """
    if not isinstance(data, dict):
        return {"hazards": [], "blocked": [], "survivor": None, "direction_bias": None, "summary": ""}

    clean = {
        "hazards": [],
        "blocked": [],
        "survivor": None,
        "direction_bias": data.get("direction_bias"),
        "summary": data.get("summary", ""),
    }

    hazards = data.get("hazards", [])
    if isinstance(hazards, list):
        for h in hazards:
            if not isinstance(h, dict):
                continue
            coord = h.get("coord")
            if not (isinstance(coord, list) and len(coord) >= 2):
                continue
            try:
                x, y = int(coord[0]), int(coord[1])
            except Exception:
                continue
            if not _in_bounds(x, y, n):
                continue

            risk = _clamp01(h.get("risk", 0.0))
            radius = h.get("radius", 0)
            try:
                radius = int(radius)
            except Exception:
                radius = 0
            radius = max(0, radius)

            clean["hazards"].append({"coord": [x, y], "risk": risk, "radius": radius})

    blocked = data.get("blocked", [])
    if isinstance(blocked, list):
        for c in blocked:
            if not (isinstance(c, list) and len(c) >= 2):
                continue
            try:
                x, y = int(c[0]), int(c[1])
            except Exception:
                continue
            if _in_bounds(x, y, n):
                clean["blocked"].append([x, y])

    surv = data.get("survivor")
    if isinstance(surv, list) and len(surv) >= 2:
        try:
            sx, sy = int(surv[0]), int(surv[1])
            if _in_bounds(sx, sy, n):
                clean["survivor"] = [sx, sy]
        except Exception:
            pass

    if clean["direction_bias"] not in [None, "north", "south", "east", "west"]:
        clean["direction_bias"] = None

    return clean


# =============================================================================
# 3) PROBLEM SOLVER
# =============================================================================
