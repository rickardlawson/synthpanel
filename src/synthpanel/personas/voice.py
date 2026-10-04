"""Personaene svarer: spørsmål, budskapstest og samtale (L4 v0).

Hver persona spilles av en språkmodell (Claude) med personaens brief som
grunnlag – livssituasjon, interesser og verdier fra panelet. Svarene er
simuleringer, ikke målinger: de viser *hvordan* ulike grupper sannsynligvis
resonnerer, og hvor de skiller seg, men tallene er ikke et utvalg.

Krever ANTHROPIC_API_KEY i miljøet (eller i .env). Modell kan overstyres
med SYNTHPANEL_MODEL.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import httpx

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"


class NoApiKey(RuntimeError):
    pass


def _model() -> str:
    return os.environ.get("SYNTHPANEL_MODEL", DEFAULT_MODEL)


def _call(system: str, messages: list[dict], max_tokens: int = 500, tools: list | None = None,
          tool_choice: dict | None = None) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise NoApiKey("ANTHROPIC_API_KEY er ikke satt (legg den i .env)")
    body = {"model": _model(), "max_tokens": max_tokens, "system": system, "messages": messages}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    r = httpx.post(API_URL, json=body, timeout=90,
                   headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    if r.status_code >= 400:
        raise RuntimeError(f"Språkmodellen svarte {r.status_code}: {r.text[:300]}")
    return r.json()


def system_prompt(brief: str) -> str:
    return f"""Du spiller en syntetisk person i et forskningspanel som skal vise hvordan ulike deler av den norske befolkningen tenker.

{brief}

Slik svarer du:
- Svar i første person, på naturlig norsk, slik denne personen ville sagt det i en samtale – kort (2–4 setninger) med mindre du blir bedt om mer.
- La livssituasjonen, vanene og verdiene over styre hva du bryr deg om, hva du vet og hvordan du reagerer. Ikke list dem opp – vis dem.
- Ikke finn på fakta som strider mot profilen. Mangler profilen noe (yrke, hobbyer), kan du gjøre rimelige antakelser som passer, men hold deg nøktern.
- Du er en vanlig person, ikke en ekspert. Det er greit å være usikker, likegyldig, skeptisk eller uenig.
- Unngå stereotypier og karikatur; folk er sammensatte.
- Ikke si at du er en AI eller en persona med mindre du blir spurt direkte om det."""


def _text(resp: dict) -> str:
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text").strip()


def ask(persona: dict, question: str) -> dict:
    resp = _call(system_prompt(persona["brief"]), [{"role": "user", "content": question}], max_tokens=400)
    return {"id": persona["id"], "svar": _text(resp)}


REACTION_TOOL = {
    "name": "reaksjon",
    "description": "Personens reaksjon på budskapet.",
    "input_schema": {
        "type": "object",
        "properties": {
            "holdning": {"type": "integer", "minimum": -2, "maximum": 2,
                         "description": "-2 svært negativ, -1 negativ, 0 nøytral/likegyldig, 1 positiv, 2 svært positiv"},
            "sitat": {"type": "string", "description": "Førsteinntrykket i første person, 1–2 setninger, slik personen ville sagt det"},
            "treffer": {"type": "string", "description": "Det som eventuelt treffer personen (kort, kan være tomt)"},
            "skurrer": {"type": "string", "description": "Det som eventuelt skurrer eller irriterer (kort, kan være tomt)"},
            "handling": {"type": "string", "enum": ["ingen", "liker", "deler", "kjøper/prøver", "protesterer"],
                         "description": "Mest sannsynlige handling etter å ha sett budskapet"},
        },
        "required": ["holdning", "sitat", "treffer", "skurrer", "handling"],
    },
}


def react(persona: dict, message: str) -> dict:
    prompt = (f"Du ser dette budskapet (i en annonse, en sak eller i sosiale medier):\n\n«{message}»\n\n"
              "Hvordan reagerer du? Svar med verktøyet `reaksjon`.")
    resp = _call(system_prompt(persona["brief"]), [{"role": "user", "content": prompt}], max_tokens=500,
                 tools=[REACTION_TOOL], tool_choice={"type": "tool", "name": "reaksjon"})
    data = next((b["input"] for b in resp.get("content", []) if b.get("type") == "tool_use"), None)
    if data is None:  # reserve: prøv å lese JSON fra tekst
        try:
            data = json.loads(_text(resp))
        except ValueError:
            data = {"holdning": 0, "sitat": _text(resp), "treffer": "", "skurrer": "", "handling": "ingen"}
    data["holdning"] = max(-2, min(2, int(data.get("holdning", 0))))
    return {"id": persona["id"], **data}


def chat(persona: dict, messages: list[dict]) -> str:
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")]
    return _text(_call(system_prompt(persona["brief"]), msgs, max_tokens=600))


def run_all(personas: list[dict], fn, arg) -> list[dict]:
    """Kjør alle personaene parallelt. Feil for én persona stopper ikke resten."""
    def one(p):
        try:
            return fn(p, arg)
        except NoApiKey:
            raise
        except Exception as e:  # noqa: BLE001
            return {"id": p["id"], "feil": str(e)[:200]}
    with ThreadPoolExecutor(max_workers=min(10, len(personas) or 1)) as ex:
        return list(ex.map(one, personas))


def summarize(personas: list[dict], reactions: list[dict]) -> dict:
    """Vektet sammendrag av budskapstesten (vekt = grupperingens andel av utvalget)."""
    share = {p["id"]: p["andel"] for p in personas}
    ok = [r for r in reactions if "holdning" in r]
    w = sum(share[r["id"]] for r in ok)
    if not ok or not w:
        return {}
    mean = sum(share[r["id"]] * r["holdning"] for r in ok) / w
    pos = sum(share[r["id"]] for r in ok if r["holdning"] > 0) / w
    neg = sum(share[r["id"]] for r in ok if r["holdning"] < 0) / w
    return {"vektet_holdning": mean, "andel_positive": pos, "andel_negative": neg, "antall_svar": len(ok)}
