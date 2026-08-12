import json
import re
import time
from datetime import datetime, timezone
from typing import Any, TypedDict
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI
from .config import get_settings
from .database import db, get_history, properties
from .grounding import evidence_answer, semantic_knowledge_search
from .nearby_transit import nearest_metro_for_property

settings = get_settings()
llm = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": settings.app_url, "X-OpenRouter-Title": settings.app_name},
)
_llm_rate_limited_until = 0.0


_NEARBY_INVENTORY_CITIES: dict[str, tuple[str, ...]] = {
    "delhi": ("Noida", "Gurugram"),
    "new delhi": ("Noida", "Gurugram"),
    "gurgaon": ("Gurugram", "Noida"),
    "bangalore": ("Bengaluru",),
}


async def _available_inventory_cities() -> list[str]:
    """Return only cities that currently have active MongoDB inventory."""
    try:
        cities = await properties.distinct("city", {"active": {"$ne": False}})
        return sorted(str(city) for city in cities if city)
    except Exception:
        return []


async def _nearby_inventory_cities(requested_city: str) -> list[str]:
    """Return configured nearby cities only when they have active inventory."""
    available_by_key = {
        available_city.casefold(): available_city
        for available_city in await _available_inventory_cities()
    }
    return [
        available_by_key[candidate.casefold()]
        for candidate in _NEARBY_INVENTORY_CITIES.get(requested_city.casefold(), ())
        if candidate.casefold() in available_by_key
    ]


async def _no_inventory_answer(requested_city: str) -> str:
    """Give an explicit no-inventory answer and grounded nearby alternatives."""
    city = requested_city.strip() or "that city"
    nearby = await _nearby_inventory_cities(city)
    opening = f"We currently don't have any properties available in {city}."
    if len(nearby) >= 2:
        return (
            f"{opening} We do have listings in {nearby[0]} and {nearby[1]}. "
            "Would you like to explore either city?"
        )
    if nearby:
        return (
            f"{opening} We do have listings in {nearby[0]}. "
            "Would you like to explore that city?"
        )
    return f"{opening} Would you like to try another city or adjust your requirements?"


async def _create_completion(**kwargs: Any) -> Any:
    """Avoid repeatedly waiting on a provider whose free quota is exhausted."""
    global _llm_rate_limited_until
    if time.monotonic() < _llm_rate_limited_until:
        raise RuntimeError("LLM temporarily unavailable after rate limit")
    try:
        return await llm.chat.completions.create(**kwargs)
    except Exception as exc:
        if getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError":
            _llm_rate_limited_until = time.monotonic() + 300
        raise


def _safe_customer_reply(reply: str) -> bool:
    """Reject provider reasoning or prompt leakage before it reaches WhatsApp."""
    value = reply.strip()
    lowered = value.casefold()
    forbidden = (
        "thinking process", "analyze user", "required message",
        "customer message", "verified context", "identify role",
        "role & constraints", "role and constraints", "system prompt",
        "chain of thought", "step-by-step reasoning", "here's my reasoning",
        "analysis:", "final answer:", "return plain text only",
        "required_message", "verified_context",
    )
    return bool(value) and len(value) <= 500 and len(value.split()) <= 70 and not any(
        marker in lowered for marker in forbidden
    ) and "```" not in value and "**" not in value


async def generate_advisor_reply(
    required_message: str,
    customer_message: str = "",
    context: dict[str, Any] | None = None,
) -> str:
    """Let the model speak naturally while preserving validated application facts."""
    prompt = """You are Aira, a concise and human Indian real-estate advisor.
Write the WhatsApp reply that fulfils REQUIRED MESSAGE using CUSTOMER MESSAGE and VERIFIED CONTEXT.
Begin by naturally acknowledging the customer's latest choice, correction, answer or concern.
Do not merely repeat their sentence. Mirror their language (English, Hindi or Hinglish) and tone.
Preserve every property name, amount, date, time, availability result and status exactly.
Never invent a listing, fact, booking or availability. Acknowledge the customer's latest correction.
Ask at most one necessary question. Do not add a consultation question unless REQUIRED MESSAGE asks for it.
Use 1 or 2 relevant, professional emojis when appropriate; never add a random emoji.
Do not restart or repeat the conversation. Use at most 45 words. Return plain text only."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    "required_message": required_message,
                    "customer_message": customer_message,
                    "verified_context": context or {},
                }, ensure_ascii=False, default=str)},
            ],
            temperature=0.25,
            max_tokens=100,
        )
        reply = str(result.choices[0].message.content or "").strip()
        return reply if _safe_customer_reply(reply) else required_message
    except Exception:
        return required_message


async def finalize_agent_answer(
    result: dict[str, Any], customer_message: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Use an existing model answer once, otherwise make exactly one writing call."""
    answer = str(result.get("answer") or "").strip()
    if result.get("answer_ready") and _safe_customer_reply(answer):
        return answer
    return await generate_advisor_reply(answer, customer_message, context or {})


async def classify_customer_control(
    customer_message: str, context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the model to distinguish opt-out from ordinary preference changes."""
    prompt = """Classify a real-estate customer's latest WhatsApp message in context.
Choose exactly one action:
- global_opt_out: they want all sales/property messages stopped or are no longer property-shopping.
- preference_change: they reject a city, budget, configuration or other criterion and provide/prefer another.
- property_rejection: they reject the displayed property but may still want other options.
- continue: any other reply.
"Not interested in Bengaluru, I want Pune" is preference_change, never global_opt_out.
Return ONLY JSON: {"action":"...","reason":"short reason"}. Do not write a reply."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    "customer_message": customer_message,
                    "conversation_context": context or {},
                }, ensure_ascii=False, default=str)},
            ],
            temperature=0,
            max_tokens=80,
        )
        data = _json_object(result.choices[0].message.content or "{}")
        action = str(data.get("action") or "").strip()
        if action not in {
            "global_opt_out", "preference_change", "property_rejection", "continue",
        }:
            return {"action": "unknown", "reason": "invalid model result"}
        return {"action": action, "reason": str(data.get("reason") or "")[:160]}
    except Exception:
        return {"action": "unknown", "reason": "model unavailable"}


async def extract_customer_identity(text: str) -> dict[str, str | None]:
    """Extract a person's stated name through the model and preserve extra intent."""
    cleaned = " ".join(text.strip().strip(".,|").split())
    explicit = re.match(
        r"(?i)^(?:my name is|i am|i'm|im|mera naam(?: hai)?|main)\s+"
        r"([a-z][a-z .'-]{0,59}?)(?:\s+(?:hai|hoon))?$",
        cleaned,
    )
    candidate = (explicit.group(1) if explicit else cleaned).strip(" .,|")
    blocked = re.search(
        r"(?i)\b(want|need|looking|property|flat|house|buy|rent|sell|book|"
        r"appointment|visit|call|email|budget|city|show|help|hello|hii?|hey)\b",
        candidate,
    )
    if (
        candidate and not blocked and 1 <= len(candidate.split()) <= 5
        and len(candidate) <= 60 and "@" not in candidate
        and not any(character.isdigit() for character in candidate)
        and re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", candidate)
    ):
        return {"name": candidate.title(), "remaining_message": ""}
    prompt = """Extract the customer's own name from their WhatsApp message.
Understand English, Hindi, Hinglish, misspellings and phrases such as "I am Devesh",
"mera naam Govind hai", or a name sent alone. Never treat a city, property, greeting,
email, phone number or request as a person's name.
Return ONLY JSON: {"name":string|null,"remaining_message":string}.
remaining_message contains any request besides the name, otherwise an empty string."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=80,
        )
        data = _json_object(result.choices[0].message.content or "{}")
        name = str(data.get("name") or "").strip(" .,|")
        if (
            not name or len(name) > 60 or len(name.split()) > 5
            or "@" in name or any(character.isdigit() for character in name)
        ):
            name = ""
        return {
            "name": name or None,
            "remaining_message": str(data.get("remaining_message") or "").strip(),
        }
    except Exception:
        candidate = re.sub(
            r"(?i)^\s*(?:my name is|i am|i'm|im|mera naam)\s+", "", text
        ).strip(" .,|")
        valid = (
            1 <= len(candidate.split()) <= 5 and len(candidate) <= 60
            and "@" not in candidate and not any(char.isdigit() for char in candidate)
        )
        return {"name": candidate if valid else None, "remaining_message": ""}


async def generate_property_captions(
    cards: list[dict[str, str]], customer_message: str,
) -> list[str]:
    """Generate one natural caption batch while locking every listing fact."""
    if not cards:
        return []
    prompt = """You write concise WhatsApp property-card captions as Aira.
Return ONLY a JSON array of strings, one caption for each VERIFIED CARD in the same order.
Each caption must contain the exact title, location, configuration, area and price supplied.
Use 2 or 3 relevant professional emojis and one short natural call to action.
Do not invent, change, compare or omit facts. Do not mention booking unless the customer asked to book."""
    fallbacks = [card["fallback"] for card in cards]
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    "customer_message": customer_message,
                    "verified_cards": [{key: value for key, value in card.items() if key != "fallback"} for card in cards],
                }, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=min(900, 180 * len(cards)),
        )
        raw = str(result.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I | re.M).strip()
        captions = json.loads(raw)
        if not isinstance(captions, list) or len(captions) != len(cards):
            return fallbacks
        checked: list[str] = []
        for caption, card, fallback in zip(captions, cards, fallbacks):
            value = str(caption).strip()
            required = (card["title"], card["location"], card["price"])
            checked.append(value if all(item in value for item in required) else fallback)
        return checked
    except Exception:
        return fallbacks


async def extract_schedule(
    text: str, saved_date: str = "", now: datetime | None = None,
) -> tuple[datetime | None, bool, bool]:
    """Understand natural scheduling language through the model, with a safe fallback."""
    from zoneinfo import ZoneInfo
    from .site_visits import parse_requested_slot

    india = ZoneInfo("Asia/Kolkata")
    local_now = (now or datetime.now(india)).astimezone(india)
    fallback_slot, fallback_has_date, fallback_has_time = parse_requested_slot(
        text, saved_date, now
    )
    # The local parser handles ordinary dates/times reliably. Use the model only
    # for wording it cannot understand, such as unusual misspellings or idioms.
    if fallback_has_date or fallback_has_time:
        return fallback_slot, fallback_has_date, fallback_has_time
    prompt = """Extract an Indian customer appointment date and time. Resolve relative phrases,
misspellings and ordinals such as tomorrow, day after tomorrow, tommorow, this Friday, or
31st this month. Combine SAVED DATE with CUSTOMER MESSAGE. Use Asia/Kolkata and CURRENT TIME.
Return ONLY JSON: {"has_date":boolean,"has_time":boolean,"local_iso":string|null}.
local_iso must include +05:30. Never invent a time when none was given."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    "current_time": local_now.isoformat(),
                    "saved_date": saved_date,
                    "customer_message": text,
                })},
            ],
            temperature=0,
            max_tokens=120,
        )
        data = _json_object(result.choices[0].message.content or "{}")
        has_date = data.get("has_date") is True
        has_time = data.get("has_time") is True
        raw = data.get("local_iso")
        slot = datetime.fromisoformat(str(raw)).astimezone(timezone.utc) if raw and has_date and has_time else None
        # The model can broaden natural-language coverage, but it cannot erase
        # a date/time that the reliable parser clearly recognized.
        if fallback_has_date and not has_date:
            return fallback_slot, fallback_has_date, fallback_has_time
        if fallback_has_time and not has_time:
            return fallback_slot, fallback_has_date, fallback_has_time
        return slot, has_date, has_time
    except Exception:
        return fallback_slot, fallback_has_date, fallback_has_time


async def classify_property_response(
    text: str, properties: list[str]
) -> dict[str, Any]:
    """Use the LLM to understand a response to a displayed property gallery."""
    if not properties:
        return {"is_interested": False, "selected_property": None}
    normalized = text.casefold()
    selected = next(
        (title for title in properties if title.casefold() in normalized), None
    )
    if selected:
        is_question = bool(re.search(
            r"\?|\b(what|where|which|how|does|is|are|can|photo|image|price|"
            r"parking|rera|available|nearby|metro|amenities)\b",
            normalized,
        ))
        return {
            "is_interested": True,
            "selected_property": selected,
            "is_question": is_question,
        }
    prompt = """You classify a user's reply after a real-estate assistant displayed listings.
Decide whether the user is expressing interest in selecting or proceeding with one of the
displayed properties. Understand natural language, misspellings, Hinglish, indirect approval,
references such as 'that one', and listing names. Do not treat a request for a new city or a
new property search as interest.

Return ONLY JSON with:
- is_interested: boolean
- selected_property: the exact property name from DISPLAYED PROPERTIES, or null if ambiguous
- is_question: boolean (true when the user asks for facts, details, comparison, or photos)

Never invent a property name."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"DISPLAYED PROPERTIES:\n{json.dumps(properties, ensure_ascii=False)}\n\n"
                        f"USER REPLY:\n{text}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=160,
        )
        decision = _json_object(result.choices[0].message.content or "{}")
    except Exception:
        selected = next(
            (title for title in properties if title.casefold() in text.casefold()), None
        )
        normalized = text.casefold()
        ordinal_words = {
            "first": 0, "1st": 0, "second": 1, "2nd": 1,
            "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4,
        }
        if not selected:
            for word, index in ordinal_words.items():
                if re.search(rf"\b{re.escape(word)}\b", normalized) and index < len(properties):
                    selected = properties[index]
                    break
        is_question = bool(re.search(
            r"\?|\b(what|where|which|how|does|is|are|photo|image|price|parking|rera)\b",
            normalized,
        ))
        expresses_interest = bool(re.search(
            r"\b(interested|like|love|choose|select|good|perfect|proceed|this one|that one)\b",
            normalized,
        ))
        return {
            "is_interested": expresses_interest and not is_question,
            "selected_property": selected,
            "is_question": bool(selected) and is_question,
        }
    selected = decision.get("selected_property")
    canonical = next(
        (title for title in properties if str(selected or "").casefold() == title.casefold()),
        None,
    )
    # If the model recognizes interest but slightly reformats the title, resolve a
    # title explicitly written by the user back to the stored canonical name.
    if not canonical:
        canonical = next(
            (title for title in properties if title.casefold() in text.casefold()),
            None,
        )
    if not canonical:
        mentioned = [
            title for title in properties
            if any(
                len(token) >= 4 and re.search(rf"\b{re.escape(token)}\b", text, re.I)
                for token in re.findall(r"[A-Za-z0-9]+", title)
            )
        ]
        canonical = mentioned[0] if len(mentioned) == 1 else None
    return {
        "is_interested": decision.get("is_interested") is True,
        "selected_property": canonical,
        "is_question": decision.get("is_question") is True or "?" in text,
    }


async def advise_on_selected_property(text: str, listing: dict[str, Any]) -> dict[str, Any]:
    """Answer freely after selection and route booking only when the user requests it."""
    normalized = " ".join(text.casefold().replace("’", "'").split())
    wants_images = bool(re.search(
        r"\b(photo|photos|image|images|gallery|picture|pictures|pic|pics|tasveer)\b",
        normalized,
    ))
    property_fact_question = bool(re.search(
        r"\b(price|cost|rate|daam|keemat|maintenance|parking|park|gaadi|area|"
        r"bedroom|bathroom|balcon|furnish|floor|facing|rera|verified|possession|"
        r"available|amenit|facility|metro|nearby|location|address|agent|owner|"
        r"photo|image|gallery|picture|pic|tasveer)\w*\b",
        normalized,
    ))
    # Keep high-confidence conversational actions deterministic. This avoids a
    # free model changing the meaning of the same booking phrase between calls.
    if re.search(
        r"\b(not ready|not now|maybe later|some other time|abhi nahi)\b.*\b(book|call|meet|visit)?\b|"
        r"\b(don't|do not|dont)\s+(?:want to\s+)?book\s+(?:yet|now)\b",
        normalized,
    ):
        return {
            "action": "decline", "wants_images": False,
            "answer": "No problem—ask anything about the property, and we can book whenever you are ready.",
        }
    if re.search(
        r"\b(cancel|withdraw|call off)\b.*\b(booking|appointment|consultation|call|visit)\b|"
        r"\b(booking|appointment|consultation|call|visit)\b.*\b(cancel|withdraw)\b",
        normalized,
    ):
        return {"action": "cancel", "wants_images": False, "answer": ""}
    if re.search(
        r"\b(book|schedule|arrange)\b.{0,35}\b(call|consultation|meeting|appointment|visit)\b|"
        r"\b(call|consultation|meeting|appointment|visit)\b.{0,35}\b(book|schedule|arrange)\b|"
        r"\b(want|would like|can we|let's|lets)\b.{0,45}\b(call|consultation|meeting|"
        r"site visit|property visit|visit the property|google meet|zoom|video)\b|"
        r"\bdiscuss\b.{0,30}\b(phone|call|video|meet|zoom)\b|"
        r"\b(?:want|like|ready)\s+to\s+book\b|\bbook\s+(?:it|this|that|now)\b",
        normalized,
    ):
        return {"action": "book", "wants_images": False, "answer": ""}
    if property_fact_question:
        return {
            "action": "question", "wants_images": wants_images,
            "answer": _database_property_answer(text, listing),
        }

    knowledge = (await retrieve_knowledge({"message": text})).get("knowledge", [])
    prompt = """You are Aira, a concise Indian real-estate advisor. The customer has already
selected a property and the assistant has offered to book a consultation.

Return ONLY JSON with:
- action: book | cancel | question | decline | other
- wants_images: boolean
- answer: a concise, helpful reply of at most 3 sentences

Use action=book only when the customer explicitly accepts or asks to book, schedule, call,
visit, or arrange a consultation. A simple positive opinion about the property is not booking.
Use action=cancel when the customer explicitly wants to cancel or withdraw the consultation.
Use action=question for questions or requests for information/photos. Understand natural
language, misspellings and Hinglish. Answer only from PROPERTY and KNOWLEDGE. If the requested
fact is absent, say it needs confirmation; never invent it. Do not ask for contact details in
the answer. For book, the answer may be empty because the application collects details."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": (
                    f"PROPERTY: {json.dumps(listing, ensure_ascii=False, default=str)}\n"
                    f"KNOWLEDGE: {json.dumps(knowledge, ensure_ascii=False, default=str)}\n"
                    f"CUSTOMER: {text}"
                )},
            ],
            temperature=0.15,
            max_tokens=350,
        )
        decision = _json_object(result.choices[0].message.content or "{}")
    except Exception:
        lowered = text.casefold()
        if re.search(r"\b(book|schedule|arrange|call|visit|appointment|meeting)\b", lowered):
            action = "book"
        elif re.search(r"\b(cancel|withdraw|stop appointment|don't book|do not book)\b", lowered):
            action = "cancel"
        elif re.search(r"\b(no thanks|not interested)\b", lowered):
            action = "decline"
        else:
            action = "question"
        return {
            "action": action,
            "wants_images": bool(re.search(r"\b(photo|image|gallery|picture)s?\b", lowered)),
            "answer": "" if action == "book" else _database_property_answer(text, listing),
        }
    action = decision.get("action")
    if action not in {"book", "cancel", "question", "decline", "other"}:
        action = "other"
    # The model classifies the action, but it is never trusted as the source of
    # property facts. Final factual text is rendered from MongoDB/KB evidence.
    grounded_answer = ""
    if action not in {"book", "cancel", "decline"}:
        grounded_answer = evidence_answer(knowledge) or _database_property_answer(text, listing)
    return {
        "action": action,
        "wants_images": decision.get("wants_images") is True or wants_images,
        "answer": grounded_answer,
    }


async def classify_booking_cancellation(text: str, booking: dict[str, Any]) -> bool:
    """Understand natural-language cancellation of an existing consultation request."""
    normalized = " ".join(text.casefold().replace("’", "'").split())
    if re.search(
        r"\b(cancel|withdraw|call off|stop)\b.*\b(appointment|booking|consultation|call|visit)\b|"
        r"\b(don't|do not|dont)\s+(?:want to\s+)?(?:book|continue|proceed)\b|"
        r"\b(?:appointment|booking|consultation|site visit)\b.*\b(cancel|withdraw|stop)\b",
        normalized,
    ):
        return True
    prompt = """Determine whether the customer explicitly wants to cancel, withdraw, or stop
the EXISTING property consultation/appointment shown below. Understand misspellings, indirect
wording and Hinglish. Do not interpret a new property search or an unrelated 'no' as a
cancellation. Return ONLY JSON: {"cancel": boolean}."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": (
                    f"EXISTING BOOKING: {json.dumps(booking, ensure_ascii=False, default=str)}\n"
                    f"CUSTOMER: {text}"
                )},
            ],
            temperature=0,
            max_tokens=80,
        )
        return _json_object(result.choices[0].message.content or "{}").get("cancel") is True
    except Exception:
        return bool(re.search(
            r"\b(cancel|withdraw|call off|stop)\b.*\b(appointment|booking|consultation|call|visit)\b|"
            r"\b(don't|do not|dont)\s+(?:want to\s+)?(?:book|continue|proceed)\b",
            normalized,
        ))


def might_cancel_booking(text: str) -> bool:
    """Cheap high-recall gate; ordinary messages must not invoke cancellation AI."""
    normalized = " ".join(text.casefold().replace("’", "'").split())
    return bool(re.search(
        r"\b(cancel|cancelled|canceled|cancellation|withdraw|call off|stop|postpone|reschedule|"
        r"can't make|cannot make|cant make|won't make|wont make|not attend|"
        r"skip (?:the )?(?:visit|call|meeting)|no longer (?:need|want))\b",
        normalized,
    ))


async def classify_booking_type(text: str) -> str | None:
    """Classify how the customer wants to meet without inventing a choice."""
    normalized = " ".join(text.casefold().replace("’", "'").split())
    if re.search(
        r"\b(site\s*visit(?:ing)?|property\s*visit(?:ing)?|physical\s*visit|visit\s+(?:the\s+)?"
        r"(?:site|property|flat|house|home)|see\s+(?:the\s+)?(?:property|flat|house|home)"
        r"|come\s+(?:and\s+)?see)\b",
        normalized,
    ):
        return "site_visit"
    if re.search(
        r"\b(phone\s*call|voice\s*call|video\s*call|google\s*meet|zoom|whatsapp\s*call|"
        r"advisor\s*(?:call|appointment)|appointment|consultation|online\s*meeting|callback|"
        r"call\s+me|speak\s+to\s+(?:an?\s+)?advisor)\b",
        normalized,
    ):
        return "appointment"
    if re.fullmatch(
        r"(?:yes[,. ]*)?(?:please[,. ]*)?(?:book|schedule|arrange)"
        r"(?:\s+(?:it|this|that|now|please))?[.! ]*",
        normalized,
    ):
        return None
    prompt = """Classify the customer's preferred real-estate booking type.
Return ONLY JSON: {"booking_type":"site_visit"|"appointment"|null}.
site_visit means physically visiting or seeing the property.
appointment means a phone/video/online consultation with an advisor.
Use null when the customer only says book/schedule/yes and has not chosen how to meet.
Understand English, Hindi, Hinglish, misspellings, and indirect wording. Never guess."""
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=80,
        )
        booking_type = _json_object(
            result.choices[0].message.content or "{}"
        ).get("booking_type")
        return booking_type if booking_type in {"site_visit", "appointment"} else None
    except Exception:
        return None


class AgentState(TypedDict, total=False):
    session_id: str
    message: str
    image: str | None
    history: list[dict]
    intent: dict[str, Any]
    matches: list[dict]
    page_info: dict[str, Any]
    knowledge: list[dict[str, Any]]
    answer: str
    answer_ready: bool
    model: str
    grounded: bool
    evidence: list[dict[str, Any]]


def _json_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group()) if match else {"intent": "general"}


def _is_greeting(text: str) -> bool:
    normalized = " ".join(text.lower().split()).strip(" .!?,")
    return normalized in {
        "hi", "hii", "hiii", "hello", "hey", "hello there",
        "good morning", "good afternoon", "good evening",
    }


def _starts_locationless_search(text: str, current_intent: dict[str, Any]) -> bool:
    """Detect a fresh property request that still needs a city."""
    if current_intent.get("city") or current_intent.get("cities"):
        return False
    normalized = " ".join(text.lower().split())
    return bool(re.search(
        r"\b(buy|purchase|rent|rental|property|properties|flat|apartment|villa|house|home)\b",
        normalized,
    )) and not bool(re.search(r"\b(next|more|show more|aur dikhao)\b", normalized))


def _is_seller_request(text: str) -> bool:
    """Recognize owners asking us to market or sell their own property."""
    normalized = " ".join(text.casefold().replace("’", "'").split())
    return bool(
        re.search(
            r"\b(?:i|we|my|our)\b.{0,35}\b(?:sell|selling|list|listing|resale)\b",
            normalized,
        )
        or re.search(
            r"\b(?:sell|selling|list|listing)\b.{0,30}\b(?:my|our)\b",
            normalized,
        )
        or re.search(
            r"\b(?:property|flat|apartment|villa|house|plot|ghar|makaan)\b"
            r".{0,25}\b(?:bechna|bechni|sell\s+karna|list\s+karna)\b",
            normalized,
        )
    )


def _is_listing_rejection(text: str) -> bool:
    normalized = " ".join(text.casefold().replace("’", "'").split()).strip(" .!?,")
    return normalized in {
        "no", "nope", "nah", "not these", "none of these", "i don't like these",
        "i dont like these", "show something else", "koi aur", "ye nahi",
    }


def _heuristic_intent(text: str) -> dict[str, Any]:
    """Reliable fallback for the high-value filters free models may omit."""
    lowered = text.lower()
    result: dict[str, Any] = {}
    cities = [
        "Mumbai", "Bengaluru", "Gurugram", "Pune", "Hyderabad", "Chennai",
        "Noida", "Goa", "Delhi", "Kolkata", "Ahmedabad",
    ]
    aliases = {
        "bangalore": "Bengaluru", "bengalore": "Bengaluru",
        "bengulore": "Bengaluru", "banglore": "Bengaluru",
        "gurgaon": "Gurugram", "bombay": "Mumbai",
    }
    found_cities: list[str] = []
    for alias, city in aliases.items():
        if alias in lowered and city not in found_cities:
            found_cities.append(city)
    for city in cities:
        if city.lower() in lowered and city not in found_cities:
            found_cities.append(city)
    if len(found_cities) == 1:
        result["city"] = found_cities[0]
    elif found_cities:
        result["cities"] = found_cities

    bhk = re.search(r"\b([1-6])\s*(?:bhk|bed(?:room)?s?)\b", lowered)
    if bhk:
        result["min_bedrooms"] = int(bhk.group(1))
        result["max_bedrooms"] = int(bhk.group(1))
    bathrooms = re.search(r"\b([1-6])\s*(?:bath|bathroom)s?\b", lowered)
    if bathrooms:
        result["min_bathrooms"] = int(bathrooms.group(1))
    educational_rent_question = bool(re.search(
        r"\b(rental yield|rent yield|yield calculation|calculate.*yield)\b", lowered
    ))
    if re.search(r"\b(rent|rental|lease)\b", lowered) and not educational_rent_question:
        result["purpose"] = "rent"
    elif re.search(r"\b(buy|purchase|sale|own)\b", lowered):
        result["purpose"] = "buy"
    for name in ("apartment", "villa", "plot"):
        if name in lowered:
            result["property_type"] = name

    area = re.search(
        r"\b(under|below|up to|upto|above|over|at least|min(?:imum)?)?\s*"
        r"([\d,.]+)\s*(?:sq\.?\s*ft|sqft|square feet)\b",
        lowered,
    )
    if area:
        area_value = int(float(area.group(2).replace(",", "")))
        if (area.group(1) or "").strip() in {"above", "over", "at least", "min", "minimum"}:
            result["min_area_sqft"] = area_value
        else:
            result["max_area_sqft"] = area_value

    amount = re.search(
        r"(?:under|below|upto|up to|max(?:imum)?(?: budget)?(?: of)?)\s*"
        r"(?:₹|rs\.?|inr|\?)?\s*([\d,.]+)\s*(crore|cr|lakh|lac|k)?",
        lowered,
    )
    if amount and not re.search(
        r"(?:sq\.?\s*ft|sqft|square feet)", lowered[amount.start():amount.end() + 15]
    ):
        value = float(amount.group(1).replace(",", ""))
        unit = amount.group(2)
        multiplier = 10_000_000 if unit in {"crore", "cr"} else (
            100_000 if unit in {"lakh", "lac"} else (1_000 if unit == "k" else 1)
        )
        result["max_price"] = int(value * multiplier)
        # In Indian residential search, a small monthly-sized ceiling with no
        # explicit buy/rent word is overwhelmingly a rental query.
        if result["max_price"] <= 200_000 and "purpose" not in result:
            result["purpose"] = "rent"
    if "unfurnished" in lowered:
        result["furnishing"] = "unfurnished"
    elif "semi-furnished" in lowered or "semi furnished" in lowered:
        result["furnishing"] = "semi-furnished"
    elif "fully furnished" in lowered or re.search(r"\bfurnished\b", lowered):
        result["furnishing"] = "fully furnished"
    if "ready to move" in lowered or "ready-to-move" in lowered:
        result["possession_status"] = "ready to move"
    elif "under construction" in lowered:
        result["possession_status"] = "under construction"
    facing = re.search(
        r"\b(north-east|north east|south-east|south east|north|south|east|west)\s+facing\b",
        lowered,
    )
    if facing:
        result["facing"] = facing.group(1).title().replace(" ", "-")
    parking = re.search(r"\b([1-5])\s*(?:car\s*)?parking(?:\s+spaces?)?\b", lowered)
    if parking:
        result["min_parking"] = int(parking.group(1))
    amenity_aliases = {
        "swimming pool": "Swimming pool", "pool": "Pool", "gym": "Gym",
        "clubhouse": "Clubhouse", "power backup": "Power backup",
        "metro": "Metro nearby", "garden": "Garden", "security": "24/7 security",
        "ev charging": "EV charging", "coworking": "Coworking lounge",
        "pet friendly": "Pet friendly",
    }
    requested_amenities = []
    for phrase, canonical in amenity_aliases.items():
        if phrase == "pool" and "swimming pool" in lowered:
            continue
        if phrase in lowered:
            requested_amenities.append(canonical)
    if requested_amenities:
        result["amenities"] = list(dict.fromkeys(requested_amenities))
    if any(word in lowered for word in (
        "photo", "photos", "image", "images", "gallery", "pictures",
        "pic", "pics", "tasveer",
    )):
        result["wants_images"] = True
    if re.search(r"\b(how many|count|number of)\b", lowered):
        result["_aggregate"] = "count"
    elif re.search(r"\b(cheapest|lowest priced?|most affordable)\b", lowered):
        result["_aggregate"] = "cheapest"
    elif re.search(r"\b(most expensive|highest priced?|costliest)\b", lowered):
        result["_aggregate"] = "most_expensive"
    generic_discovery = bool(re.search(
        r"\b(property|properties|flat|apartment|villa|house|home|ghar|makaan)\b",
        lowered,
    )) and bool(re.search(
        r"\b(want|need|looking|show|chahiye|chaiye|dikhao|dikhana|lena|kharidna)\b",
        lowered,
    ))
    if _is_seller_request(text):
        result["intent"] = "seller_intake"
        result["purpose"] = "sell"
    elif generic_discovery:
        result["intent"] = "property_search"
    if result and "intent" not in result:
        result["intent"] = "property_search"
    return result


def _explicitly_excluded_cities(text: str) -> list[str]:
    """Validate that an LLM never returns a city the customer just rejected."""
    aliases = {
        "mumbai": "Mumbai", "bombay": "Mumbai", "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru", "banglore": "Bengaluru", "delhi": "Delhi",
        "pune": "Pune", "gurugram": "Gurugram", "gurgaon": "Gurugram",
        "noida": "Noida", "hyderabad": "Hyderabad", "chennai": "Chennai",
    }
    lowered = " ".join(text.casefold().replace("’", "'").split())
    excluded: list[str] = []
    for alias, canonical in aliases.items():
        patterns = (
            rf"\b(?:do not|don't|dont|not|exclude|except|remove)\b[^,.!?]{{0,24}}\b{re.escape(alias)}\b",
            rf"\b(?:no|not)\s+{re.escape(alias)}\b",
            rf"\b{re.escape(alias)}\b.{{0,18}}\b(?:not|nahi|mat)\b",
        )
        if any(re.search(pattern, lowered) for pattern in patterns) and canonical not in excluded:
            excluded.append(canonical)
    return excluded


async def load_context(state: AgentState) -> dict:
    return {"history": await get_history(state["session_id"], settings.max_history_messages)}


async def understand(state: AgentState) -> dict:
    if _is_greeting(state["message"]):
        return {"intent": {"intent": "greeting"}, "model": "deterministic"}
    if _is_listing_rejection(state["message"]):
        return {"intent": {"intent": "listing_rejection"}, "model": "deterministic"}
    current_reliable = _heuristic_intent(state["message"])
    normalized_message = state["message"].casefold()
    database_question = bool(current_reliable) or bool(re.search(
        r"\b(price|cost|maintenance|parking|area|bhk|bedroom|bathroom|balcon|"
        r"furnish|floor|facing|rera|verified|possession|available|amenit|nearby|"
        r"agent|photo|image|compare|cheapest|expensive)\b",
        normalized_message,
    ))
    knowledge_question = bool(re.search(
        r"\b(document|due diligence|home loan|mortgage|emi|site visit|inspection|"
        r"rental yield|roi|investment|valuation|selling process|sell property)\b",
        normalized_message,
    ))
    prompt = """You are the query-understanding layer of an Indian real-estate assistant.
Return ONLY a JSON object with: intent (property_search|seller_intake|compare|valuation|image_analysis|general),
city, cities (array), excluded_cities (array), locality, property_type, purpose (buy|rent|sell), min_price, max_price, min_bedrooms,
max_bedrooms, min_bathrooms, min_area_sqft, max_area_sqft, furnishing, possession_status,
facing, min_parking, amenities (array), wants_images (boolean), and user_language. Use null when unknown.
Set wants_images true when the user asks to see photos, pictures, images, interiors, or a gallery.
Resolve follow-up references from conversation. Treat corrections such as "not Mumbai, Bangalore"
as city=Bengaluru and excluded_cities=[Mumbai]. Return CURRENT active criteria, not every city
ever mentioned. INR amounts: lakh=100000, crore=10000000."""
    content: list[dict] = [{"type": "text", "text": state["message"]}]
    if state.get("image"):
        content.append({"type": "image_url", "image_url": {"url": state["image"]}})
    prior = [{"role": x["role"], "content": x["content"]} for x in state["history"][-8:]]
    model_name = "deterministic"
    deterministic_route = bool(current_reliable) and not state.get("image") and not (
        _explicitly_excluded_cities(state["message"])
    )
    if deterministic_route:
        model_intent = dict(current_reliable)
        model_name = "deterministic"
    else:
        try:
            result = await _create_completion(
                model=settings.openrouter_model,
                messages=[{"role": "system", "content": prompt}, *prior, {"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=500,
            )
            model_intent = _json_object(result.choices[0].message.content or "{}")
            model_name = result.model
        except Exception:
            # Availability fallback only; MongoDB remains the source of facts.
            model_intent = {"intent": "general"}
            model_name = "deterministic-fallback"
    # Carry deterministic filters from earlier user turns into short follow-ups
    # such as "what about Gurugram?", then let the current message override them.
    previous_text = " ".join(
        item["content"] for item in state["history"][-8:] if item.get("role") == "user"
    )
    reliable = _heuristic_intent(previous_text)
    fresh_locationless_search = _starts_locationless_search(
        state["message"], current_reliable
    ) and not _is_seller_request(previous_text)
    validated_exclusions = list(dict.fromkeys([
        *model_intent.get("excluded_cities", []),
        *_explicitly_excluded_cities(state["message"]),
    ]))
    model_intent["excluded_cities"] = validated_exclusions
    model_excluded = {str(city).casefold() for city in validated_exclusions}
    current_locations = current_reliable.get("cities") or (
        [current_reliable.get("city")] if current_reliable.get("city") else []
    )
    positive_current_location = any(
        str(city).casefold() not in model_excluded for city in current_locations
    )
    if current_locations and positive_current_location:
        reliable.pop("city", None)
        reliable.pop("cities", None)
    elif fresh_locationless_search:
        # Do not let an old search location silently answer a new request.
        reliable.pop("city", None)
        reliable.pop("cities", None)
        reliable.pop("locality", None)
        model_intent.pop("city", None)
        model_intent.pop("cities", None)
        model_intent.pop("locality", None)
    current_reliable_locations_removed = dict(current_reliable)
    if current_locations and not positive_current_location:
        current_reliable_locations_removed.pop("city", None)
        current_reliable_locations_removed.pop("cities", None)
    reliable.update(current_reliable_locations_removed)
    next_page = bool(re.search(
        r"\b(next|more|next properties|show more|aur dikhao|agla|agli)\b",
        state["message"].lower(),
    ))
    intent = {**model_intent, **reliable}
    excluded = {str(city).casefold() for city in model_intent.get("excluded_cities", [])}
    if excluded:
        active = intent.get("cities") or ([intent.get("city")] if intent.get("city") else [])
        active = [city for city in active if str(city).casefold() not in excluded]
        intent.pop("city", None)
        intent.pop("cities", None)
        if len(active) == 1:
            intent["city"] = active[0]
        elif active:
            intent["cities"] = active
        intent["excluded_cities"] = list(model_intent.get("excluded_cities", []))
    seller_context = _is_seller_request(state["message"]) or _is_seller_request(previous_text)
    explicit_buyer_request = bool(re.search(
        r"\b(?:buy|purchase|rent|looking\s+to\s+(?:buy|rent))\b",
        state["message"].casefold(),
    ))
    if seller_context and not explicit_buyer_request:
        intent["intent"] = "seller_intake"
        intent["purpose"] = "sell"
    intent["_current_filters"] = current_reliable
    intent["_next_page"] = next_page
    if next_page:
        intent["intent"] = "property_search"
    return {"intent": intent, "model": model_name}


async def search_inventory(state: AgentState) -> dict:
    data = dict(state["intent"])
    if data.get("intent") not in {"property_search", "compare"}:
        return {"matches": [], "page_info": {}}

    session_id = state["session_id"]
    previous = await db.search_sessions.find_one({"session_id": session_id})
    next_page = bool(data.pop("_next_page", False))
    nearby_fallback = bool(next_page and previous and previous.get("nearby_fallback"))
    requested_city = str(previous.get("requested_city") or "") if nearby_fallback else ""
    current_filters = data.pop("_current_filters", {}) or {}
    aggregate = data.get("_aggregate")
    if next_page and previous:
        data = {**previous.get("criteria", {}), **current_filters}
        data["intent"] = "property_search"
    criteria = {
        key: value for key, value in data.items()
        if value not in (None, "", [], {}) and not key.startswith("_")
    }
    cities = criteria.get("cities") or (
        [criteria["city"]] if criteria.get("city") else []
    )
    if isinstance(cities, str):
        cities = [cities]
    cities = list(dict.fromkeys(str(city) for city in cities))

    message = str(state.get("message") or "")
    explicit_discovery = bool(re.search(
        r"\b(chahiye|chaiye|looking\s+for|want\s+to\s+(?:buy|rent)|"
        r"show\s+(?:me\s+)?(?:some\s+)?propert|properties?\s+in)\b",
        message, re.I,
    ))
    if not next_page and (
        cities or criteria.get("locality") or explicit_discovery
        or not _looks_like_property_followup(message)
    ):
        # Clear an earlier selection only for a genuine new discovery. Some
        # short property follow-ups are conservatively classified as searches
        # by the model, but still need their selected-property context.
        await db.search_sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "session_id": session_id,
                "selected_property": None,
                "updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    # Buying/renting without a location is a discovery request, not permission
    # to dump inventory from every market in the database.
    if not cities and not criteria.get("locality") and aggregate != "count":
        return {
            "matches": [],
            "page_info": {"requires_city": True, "cities": [], "total": 0},
        }

    query: dict[str, Any] = {"active": {"$ne": False}}
    for source, target in [
        ("locality", "locality"), ("property_type", "type"), ("purpose", "purpose"),
        ("furnishing", "furnishing"), ("possession_status", "possession_status"),
        ("facing", "facing"),
    ]:
        if criteria.get(source):
            query[target] = {"$regex": f"^{re.escape(str(criteria[source]))}$", "$options": "i"}
    price = {}
    if isinstance(criteria.get("min_price"), (int, float)): price["$gte"] = criteria["min_price"]
    if isinstance(criteria.get("max_price"), (int, float)): price["$lte"] = criteria["max_price"]
    if price: query["price"] = price
    if isinstance(criteria.get("min_bedrooms"), (int, float)):
        query["bedrooms"] = {"$gte": criteria["min_bedrooms"]}
    if isinstance(criteria.get("max_bedrooms"), (int, float)):
        query.setdefault("bedrooms", {})["$lte"] = criteria["max_bedrooms"]
    if isinstance(criteria.get("min_bathrooms"), (int, float)):
        query["bathrooms"] = {"$gte": criteria["min_bathrooms"]}
    area_query = {}
    if isinstance(criteria.get("min_area_sqft"), (int, float)):
        area_query["$gte"] = criteria["min_area_sqft"]
    if isinstance(criteria.get("max_area_sqft"), (int, float)):
        area_query["$lte"] = criteria["max_area_sqft"]
    if area_query:
        query["area_sqft"] = area_query
    if isinstance(criteria.get("min_parking"), (int, float)):
        query["parking_spaces"] = {"$gte": criteria["min_parking"]}
    if criteria.get("amenities"):
        # MongoDB does not allow {$regex: ...} expression documents inside
        # $all. Match each requested amenity against the array in a separate
        # $and clause instead. A regex applied to an array field matches any
        # element, while $and preserves the required "all amenities" meaning.
        query.setdefault("$and", []).extend(
            {
                "amenities": {
                    "$regex": re.escape(str(amenity)),
                    "$options": "i",
                }
            }
            for amenity in criteria["amenities"][:3]
        )

    if aggregate == "count":
        per_city_total: dict[str, int] = {}
        if len(cities) >= 2:
            for city in cities:
                city_query = {
                    **query,
                    "city": {"$regex": f"^{re.escape(city)}$", "$options": "i"},
                }
                per_city_total[city] = await properties.count_documents(city_query)
            total = sum(per_city_total.values())
        else:
            if cities:
                query["city"] = {
                    "$regex": f"^{re.escape(cities[0])}$", "$options": "i"
                }
            total = await properties.count_documents(query)
        return {
            "matches": [],
            "page_info": {
                "count_query": True, "total": total, "cities": cities,
                "per_city_total": per_city_total, "exhausted": False,
            },
        }

    offset = int(previous.get("next_offset", 0)) if next_page and previous else 0
    matches: list[dict] = []
    per_city_shown: dict[str, int] = {}
    per_city_total: dict[str, int] = {}
    if len(cities) >= 2 and nearby_fallback:
        page_size = 5
        nearby_query = {
            **query,
            "$or": [
                {"city": {"$regex": f"^{re.escape(city)}$", "$options": "i"}}
                for city in cities
            ],
        }
        total = await properties.count_documents(nearby_query)
        cursor = (
            properties.find(nearby_query, {"_id": 0})
            .sort("price", -1 if aggregate == "most_expensive" else 1)
            .skip(offset).limit(page_size)
        )
        matches = [item async for item in cursor]
        per_city_shown = {
            city: sum(
                1 for item in matches
                if str(item.get("city") or "").casefold() == city.casefold()
            )
            for city in cities
        }
        per_city_total = {
            city: await properties.count_documents({
                **query,
                "city": {"$regex": f"^{re.escape(city)}$", "$options": "i"},
            })
            for city in cities
        }
    elif len(cities) >= 2:
        page_size = 1 if aggregate in {"cheapest", "most_expensive"} else 3
        for city in cities:
            city_query = {
                **query,
                "city": {"$regex": f"^{re.escape(city)}$", "$options": "i"},
            }
            per_city_total[city] = await properties.count_documents(city_query)
            cursor = (
                properties.find(city_query, {"_id": 0})
                .sort("price", -1 if aggregate == "most_expensive" else 1)
                .skip(offset).limit(page_size)
            )
            city_matches = [item async for item in cursor]
            matches.extend(city_matches)
            per_city_shown[city] = len(city_matches)
        total = sum(per_city_total.values())
    else:
        page_size = 1 if aggregate in {"cheapest", "most_expensive"} else 5
        if cities:
            query["city"] = {
                "$regex": f"^{re.escape(cities[0])}$", "$options": "i"
            }
        total = await properties.count_documents(query)
        cursor = (
            properties.find(query, {"_id": 0})
            .sort("price", -1 if aggregate == "most_expensive" else 1)
            .skip(offset).limit(page_size)
        )
        matches = [item async for item in cursor]

    # If the requested city has no matching inventory, automatically show
    # genuine listings from configured nearby cities. Every suggested city is
    # checked against the live MongoDB inventory first.
    if (
        not matches and not next_page and len(cities) == 1 and offset == 0
    ):
        requested_city = cities[0]
        nearby_cities = await _nearby_inventory_cities(requested_city)
        if nearby_cities:
            nearby_fallback = True
            cities = nearby_cities
            page_size = 5
            base_query = {key: value for key, value in query.items() if key != "city"}
            nearby_query = {
                **base_query,
                "$or": [
                    {"city": {"$regex": f"^{re.escape(city)}$", "$options": "i"}}
                    for city in cities
                ],
            }
            total = await properties.count_documents(nearby_query)
            cursor = (
                properties.find(nearby_query, {"_id": 0})
                .sort("price", -1 if aggregate == "most_expensive" else 1)
                .limit(page_size)
            )
            matches = [item async for item in cursor]
            per_city_shown = {
                city: sum(
                    1 for item in matches
                    if str(item.get("city") or "").casefold() == city.casefold()
                )
                for city in cities
            }
            per_city_total = {
                city: await properties.count_documents({
                    **base_query,
                    "city": {"$regex": f"^{re.escape(city)}$", "$options": "i"},
                })
                for city in cities
            }
            criteria.pop("city", None)
            criteria["cities"] = cities

    await db.search_sessions.update_one(
        {"session_id": session_id},
        {"$set": {
            "session_id": session_id,
            "criteria": criteria,
            "next_offset": offset + page_size,
            "nearby_fallback": nearby_fallback,
            "requested_city": requested_city or None,
            "updated_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )
    return {
        "matches": matches,
        "page_info": {
            "offset": offset,
            "page_size": page_size,
            "total": total,
            "cities": cities,
            "per_city_shown": per_city_shown,
            "per_city_total": per_city_total,
            "exhausted": not matches,
            "nearby_fallback": nearby_fallback,
            "requested_city": requested_city or None,
        },
    }


async def retrieve_knowledge(state: AgentState) -> dict:
    """Retrieve business knowledge relevant to the current question from MongoDB."""
    intent_context = " ".join(
        str(value) for key, value in state.get("intent", {}).items()
        if key in {"city", "cities", "locality", "property_type", "purpose"}
        and value not in (None, "", [])
    )
    query_text = " ".join(
        part for part in (state["message"].strip(), intent_context)
        if part
    )[-4000:]
    if not query_text:
        return {"knowledge": []}
    documents = await semantic_knowledge_search(query_text, limit=5)
    projection = {
        "_id": 0, "title": 1, "content": 1, "category": 1,
        "source": 1, "keywords": 1, "score": {"$meta": "textScore"},
    }
    if not documents:
        try:
            cursor = db.knowledge_base.find(
                {"active": {"$ne": False}, "$text": {"$search": query_text}},
                projection,
            ).sort([("score", {"$meta": "textScore"})]).limit(5)
            documents = [document async for document in cursor]
            for document in documents:
                document["retrieval_method"] = "mongodb_text"
        except Exception:
            # Supports local/test databases before their text index is ready.
            tokens = [
                token for token in re.findall(r"[A-Za-z0-9]+", query_text.lower())
                if len(token) >= 3
            ][:8]
            if tokens:
                pattern = "|".join(re.escape(token) for token in tokens)
                cursor = db.knowledge_base.find(
                    {
                        "active": {"$ne": False},
                        "$or": [
                            {"title": {"$regex": pattern, "$options": "i"}},
                            {"content": {"$regex": pattern, "$options": "i"}},
                            {"keywords": {"$regex": pattern, "$options": "i"}},
                        ],
                    },
                    {key: value for key, value in projection.items() if key != "score"},
                ).limit(5)
                documents = [document async for document in cursor]
                for document in documents:
                    document["retrieval_method"] = "lexical_fallback"
    result = {"knowledge": documents}
    if state.get("matches"):
        return result

    session_id = state.get("session_id", "")
    pending = await db.pending_leads.find_one(
        {"session_id": session_id, "selected_property": {"$nin": [None, ""]}},
        {"_id": 0, "selected_property": 1},
    )
    selected_title = pending.get("selected_property") if pending else None
    if not selected_title:
        lead_query: dict[str, Any] = {"property_title": {"$nin": [None, ""]}}
        if session_id.startswith("wa:"):
            phone = session_id.removeprefix("wa:")
            lead_query["$or"] = [{"whatsapp_phone": phone}, {"phone": phone}]
        else:
            lead_query["session_id"] = session_id
        lead = await db.leads.find_one(
            lead_query, {"_id": 0, "property_title": 1}, sort=[("updated_at", -1)]
        )
        selected_title = lead.get("property_title") if lead else None
    if selected_title:
        listing = await properties.find_one({"title": selected_title}, {"_id": 0})
        if listing:
            result["matches"] = [listing]
            result["page_info"] = {"focused_property": True, "total": 1}
    return result


_PROPERTY_REFERENCE_STOPWORDS = {
    "property", "properties", "project", "flat", "apartment", "villa", "house",
    "show", "tell", "about", "this", "that", "have", "does", "what", "where",
    "which", "with", "more", "details", "image", "images", "photo", "photos",
    "available", "interested", "looks", "good", "want", "need", "please",
    "compare", "comparison", "versus", "and", "between",
    # Question grammar, pronouns and property attributes are not part of a
    # listing title. Excluding them lets a question such as "price and parking
    # of Aravalli Heights" resolve the authoritative row by its actual name.
    "the", "its", "for", "from", "into", "our", "your", "their", "there",
    "price", "cost", "budget", "area", "size", "parking", "amenity",
    "amenities", "facility", "facilities", "location", "located", "address",
    "bedroom", "bedrooms", "bathroom", "bathrooms", "balcony", "balconies",
    "furnished", "furnishing", "floor", "facing", "maintenance", "possession",
    "verified", "rera", "agent", "contact", "nearby", "availability",
}


def _looks_like_property_followup(text: str) -> bool:
    """Recognize natural follow-ups after one property has been selected."""
    return bool(re.search(
        r"\b(its?|this|that|iska|iski|iske|uska|uski|yeh?|property|"
        r"price|cost|rate|daam|keemat|kitn[ae]|photo|photos|pic|pics|image|images|"
        r"tasveer|parking|park|gaadi|metro|subway|station|door|distance|"
        r"area|size|amenit|facility|facing|floor|rera|nearby)\b",
        text.casefold(),
    ))


def _reference_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(
        token for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) >= 3 and token not in _PROPERTY_REFERENCE_STOPWORDS
    ))


def _best_referenced_title(text: str, titles: list[str]) -> str | None:
    """Resolve natural property references without allowing the LLM to invent a title."""
    normalized = " ".join(text.casefold().split())
    direct = next((title for title in titles if title.casefold() in normalized), None)
    if direct:
        return direct
    query_tokens = set(_reference_tokens(text))
    ranked: list[tuple[float, str]] = []
    for title in titles:
        title_tokens = set(_reference_tokens(title))
        overlap = len(query_tokens & title_tokens)
        if overlap >= 2 or (overlap >= 1 and any(token.isdigit() for token in query_tokens)):
            ranked.append((overlap / max(1, len(title_tokens)), title))
    ranked.sort(reverse=True)
    return ranked[0][1] if ranked else None


def _referenced_titles(text: str, titles: list[str]) -> list[str]:
    normalized = " ".join(text.casefold().split())
    direct = list(dict.fromkeys(
        title for title in titles if title.casefold() in normalized
    ))
    if direct:
        return direct
    best = _best_referenced_title(text, titles)
    return [best] if best else []


async def resolve_property_context(state: AgentState) -> dict:
    """Fetch the exact MongoDB property row referenced by any customer question."""
    matches = state.get("matches") or []
    if len(matches) == 1:
        title = str(matches[0].get("title") or "").strip()
        if title:
            await db.search_sessions.update_one(
                {"session_id": state.get("session_id", "")},
                {"$set": {
                    "session_id": state.get("session_id", ""),
                    "selected_property": title,
                    "updated_at": datetime.now(timezone.utc),
                }},
                upsert=True,
            )
        return {}
    session_id = state.get("session_id", "")
    pending = await db.pending_leads.find_one(
        {"session_id": session_id},
        {"_id": 0, "selected_property": 1, "matched_properties": 1},
    ) or {}
    search_context = await db.search_sessions.find_one({"session_id": session_id}) or {}
    titles = [
        str(title) for title in [
            pending.get("selected_property"), search_context.get("selected_property"),
            *(pending.get("matched_properties") or [])
        ] if title
    ]
    titles = list(dict.fromkeys(titles))
    history_text = " ".join(
        str(item.get("content") or "") for item in state.get("history", [])[-6:]
        if item.get("role") == "user"
    )
    reference_text = f"{state.get('message', '')} {history_text}".strip()
    selected_titles = _referenced_titles(reference_text, titles)
    if (
        not selected_titles
        and len(titles) == 1
        and _looks_like_property_followup(state.get("message", ""))
    ):
        selected_titles = titles

    if not selected_titles:
        # Include recent user history so natural pronouns such as "its price"
        # and "show its photos" retain the last explicitly named property.
        tokens = _reference_tokens(reference_text)
        # A generic city/search question must not be converted into a random
        # property. Require a distinctive multi-token name or its numeric suffix.
        if len(tokens) >= 2 or any(token.isdigit() for token in tokens):
            # Retrieve a bounded candidate set using OR, then require a strong
            # token overlap in _referenced_titles. AND cannot resolve two
            # different titles in a comparison and attributes can otherwise
            # accidentally make an exact property name unsearchable.
            query = {"$or": [
                {"title": {"$regex": rf"\b{re.escape(token)}\b", "$options": "i"}}
                for token in tokens[:10]
            ]}
            cursor = properties.find(query, {"_id": 0, "title": 1}).limit(50)
            database_titles = [str(row["title"]) async for row in cursor if row.get("title")]
            selected_titles = _referenced_titles(reference_text, database_titles)

    if not selected_titles:
        return {}
    listings = []
    for selected_title in selected_titles[:4]:
        listing = await properties.find_one({"title": selected_title}, {"_id": 0})
        if listing:
            listings.append(listing)
    if not listings:
        return {}
    if len(listings) == 1:
        await db.search_sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "session_id": session_id,
                "selected_property": selected_titles[0],
                "updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
    intent = dict(state.get("intent") or {})
    intent["intent"] = "compare" if len(listings) > 1 else "property_question"
    intent["referenced_property"] = selected_titles[0]
    intent["referenced_properties"] = selected_titles
    return {
        "intent": intent,
        "matches": listings,
        "page_info": {
            "focused_property": True, "comparison": len(listings) > 1,
            "total": len(listings),
        },
    }


async def enrich_nearby_transit(state: AgentState) -> dict:
    """Fetch metro data only when a customer asks about nearby rail transit."""
    if not re.search(
        r"\b(metro|subway|nearest\s+(?:train\s+)?station|"
        r"station.{0,20}(?:door|distance|near))\b",
        state.get("message", ""), re.I,
    ):
        return {}
    matches = state.get("matches") or []
    if len(matches) != 1:
        return {}
    listing = dict(matches[0])
    try:
        listing["nearby_metro"] = await nearest_metro_for_property(listing)
    except RuntimeError:
        listing["nearby_metro"] = {"status": "lookup_unavailable"}
    intent = dict(state.get("intent") or {})
    intent["intent"] = "property_question"
    return {
        "intent": intent,
        "matches": [listing],
        "page_info": {**(state.get("page_info") or {}), "focused_property": True, "total": 1},
    }


def _display_money(value: Any, currency: str = "INR", monthly: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "not recorded"
    prefix = "₹" if currency.upper() == "INR" else f"{currency.upper()} "
    suffix = "/month" if monthly else ""
    return f"{prefix}{value:,.0f}{suffix}"


def _database_property_answer(message: str, listing: dict[str, Any]) -> str:
    """Answer common property questions directly from one authoritative DB row."""
    text = message.casefold()
    facts: list[str] = []
    title = str(listing.get("title") or "This property")

    def add(label: str, value: Any, *, false_text: str | None = None) -> None:
        if value is None:
            facts.append(f"{label}: not recorded")
        elif isinstance(value, bool):
            facts.append(f"{label}: {'Yes' if value else (false_text or 'No')}")
        elif isinstance(value, list):
            facts.append(f"{label}: {', '.join(map(str, value)) if value else 'not recorded'}")
        else:
            facts.append(f"{label}: {value}")

    if re.search(
        r"\b(price|cost|budget|how much|rate|daam|keemat)\b|"
        r"\bkitn[ae]\s+(?:ka|ki|ke|daam|rate|price|cost)\b",
        text,
    ):
        purpose = str(listing.get("purpose") or "")
        add("Price", _display_money(
            listing.get("price"), str(listing.get("currency") or "INR"), purpose == "rent"
        ))
    if re.search(r"\b(maintenance|monthly charge|society charge)\b", text):
        add("Monthly maintenance", _display_money(listing.get("maintenance_monthly")))
    if re.search(r"\b(parking|car park|park|gaadi)\b", text):
        add("Parking spaces", listing.get("parking_spaces"))
    if re.search(r"\b(carpet|built.?up|area|sq\s*ft|size)\b", text):
        if "carpet" in text:
            value = listing.get("carpet_area_sqft")
            add("Carpet area", f"{value:,} sq ft" if isinstance(value, (int, float)) else None)
        else:
            value = listing.get("area_sqft")
            add("Area", f"{value:,} sq ft" if isinstance(value, (int, float)) else None)
    if re.search(r"\b(bhk|bedroom|bedrooms)\b", text):
        add("Bedrooms", listing.get("bedrooms"))
    if re.search(r"\b(bath|bathroom|bathrooms)\b", text):
        add("Bathrooms", listing.get("bathrooms"))
    if re.search(r"\b(balcony|balconies)\b", text):
        add("Balconies", listing.get("balconies"))
    if re.search(r"\b(furnish|furnished|furniture)\b", text):
        add("Furnishing", listing.get("furnishing"))
    if re.search(r"\b(floor|storey|story)\b", text):
        floor = listing.get("floor")
        total = listing.get("total_floors")
        add("Floor", f"{floor} of {total}" if floor is not None and total is not None else floor)
    if re.search(r"\b(facing|east|west|north|south)\b", text):
        add("Facing", listing.get("facing"))
    if re.search(r"\b(rera|registered)\b", text):
        add("RERA registered", listing.get("rera_registered"))
    if re.search(r"\b(verified|verification)\b", text):
        add("Listing verified", listing.get("verified"))
    if re.search(r"\b(possession|ready to move|construction|available|availability)\b", text):
        add("Possession", listing.get("possession_status"))
        if listing.get("available_from") is not None:
            add("Available from", listing.get("available_from"))
    if re.search(r"\b(age|how old)\b", text):
        age = listing.get("age_years")
        add("Property age", f"{age} years" if isinstance(age, (int, float)) else None)
    if re.search(r"\b(negotiable|negotiate|discount)\b", text):
        add("Price negotiable", listing.get("price_negotiable"))
    if re.search(r"\b(amenity|amenities|facility|facilities)\b", text):
        add("Amenities", listing.get("amenities"))
    if re.search(
        r"\b(metro|subway|nearest\s+(?:train\s+)?station|"
        r"station.{0,20}(?:door|distance|near))\b",
        text,
    ):
        metro = listing.get("nearby_metro") or {}
        status = metro.get("status") if isinstance(metro, dict) else None
        if status == "found":
            distance = metro.get("distance_km")
            network = f", {metro['network']}" if metro.get("network") else ""
            facts.append(
                f"Nearest metro: {metro.get('name')}{network} — approximately "
                f"{distance:g} km straight-line (OpenStreetMap)"
            )
        elif status == "coordinates_missing":
            facts.append("Nearest metro: unavailable because this listing has no coordinates")
        elif status == "not_found":
            radius = metro.get("search_radius_km") or 10
            facts.append(f"Nearest metro: none found within {radius:g} km in OpenStreetMap")
        elif status == "lookup_unavailable":
            facts.append("Nearest metro: lookup service is temporarily unavailable")
        else:
            facts.append("Nearest metro: not checked yet")
    elif re.search(r"\b(nearby|near|school|hospital|transport)\b", text):
        add("Nearby", listing.get("nearby"))
    if re.search(r"\b(location|located|address|where)\b", text):
        location = ", ".join(filter(None, [listing.get("locality"), listing.get("city")]))
        add("Location", location or None)
    if re.search(r"\b(agent|owner|contact|phone|email)\b", text):
        agent = listing.get("agent") or {}
        add("Agent", agent.get("name") if isinstance(agent, dict) else agent)
    if re.search(
        r"\b(photo|photos|image|images|gallery|picture|pictures|pic|pics|tasveer)\b",
        text,
    ):
        count = len(listing.get("images") or [])
        facts.append(f"{count} image{'s' if count != 1 else ''} are available for {title}")

    if not facts and re.search(r"\?|\b(what|when|where|who|why|how|does|is|are|can)\b", text):
        return f"That information is not recorded for {title}. Please confirm it with the assigned property expert."
    if not facts:
        location = ", ".join(filter(None, [listing.get("locality"), listing.get("city")]))
        facts = [
            f"{listing.get('bedrooms', 'BHK not recorded')} BHK {listing.get('type', 'property')}",
            location or "location not recorded",
            _display_money(listing.get("price"), str(listing.get("currency") or "INR"),
                           listing.get("purpose") == "rent"),
        ]
    return f"{title} — " + "; ".join(facts) + "."


def _offline_answer(state: AgentState) -> str:
    matches = state.get("matches") or []
    if matches:
        if len(matches) == 1:
            return _database_property_answer(state.get("message", ""), matches[0])
        summaries = []
        for item in matches[:4]:
            location = ", ".join(filter(None, [item.get("locality"), item.get("city")]))
            summaries.append(
                f"{item.get('title')}: {item.get('bedrooms', 'BHK not recorded')} BHK, "
                f"{location or 'location not recorded'}, "
                f"{_display_money(item.get('price'), str(item.get('currency') or 'INR'), item.get('purpose') == 'rent')}"
            )
        return "Comparison from the property database — " + "; ".join(summaries) + "."
    knowledge = state.get("knowledge") or []
    if knowledge:
        answer = evidence_answer(knowledge)
        if answer:
            return answer
    if settings.strict_grounding:
        return (
            "I don't have verified information for that in the property database or "
            "knowledge base. Please ask about a listed property, or let me arrange confirmation."
        )
    return "I can help with property search, pricing, features, documents, or appointments. Which property or city should I check?"


async def compose_answer(state: AgentState) -> dict:
    # Search results are rendered as atomic image + data cards by the web client
    # and as image captions by WhatsApp. Do not ask an unpredictable free model
    # to repeat the same inventory as a long numbered Markdown list.
    if state["intent"].get("intent") == "greeting":
        return {
            "answer": "Hi! I'm Aira, your property advisor. Which city are you looking to buy or rent in?",
            "model": "deterministic",
        }
    if state["intent"].get("intent") == "listing_rejection":
        return {
            "answer": (
                "No problem. What should I change for the next options—location, "
                "budget, or property type?"
            ),
            "model": "deterministic",
        }
    if state["intent"].get("intent") == "seller_intake":
        intent = state["intent"]
        city = intent.get("city") or intent.get("locality")
        property_type = intent.get("property_type")
        bedrooms = intent.get("min_bedrooms")
        area = intent.get("max_area_sqft") or intent.get("min_area_sqft")
        expected_price = intent.get("max_price") or intent.get("min_price")
        if not city:
            answer = "Absolutely—I can help you sell it. Which city and locality is the property in?"
        elif not property_type:
            answer = f"Got it—your property is in {city}. Is it an apartment, villa, house, or plot?"
        elif not area and property_type != "plot":
            bhk_text = f"{bedrooms} BHK " if bedrooms else ""
            answer = f"Thanks. What is the approximate carpet or built-up area of the {bhk_text}{property_type}?"
        elif not expected_price:
            answer = "What selling price are you expecting? An approximate amount is fine."
        else:
            answer = (
                "Thanks—I have the basic property details. Would you like an advisor to "
                "call you for valuation and listing next steps?"
            )
        return {"answer": answer, "model": "deterministic"}
    page_info = state.get("page_info") or {}
    focused_property = bool(page_info.get("focused_property"))
    if page_info.get("count_query"):
        total = int(page_info.get("total") or 0)
        cities = page_info.get("cities") or []
        per_city = page_info.get("per_city_total") or {}
        if len(cities) >= 2:
            detail = " and ".join(f"{per_city.get(city, 0)} in {city}" for city in cities)
            answer = f"There are {total} matching properties: {detail}."
        elif cities:
            answer = f"There are {total} matching properties in {cities[0]}."
        else:
            answer = f"There are {total} active properties in our database."
        return {"answer": answer, "model": "database"}
    aggregate = state["intent"].get("_aggregate")
    if state.get("matches") and aggregate in {"cheapest", "most_expensive"}:
        listing = state["matches"][0]
        label = "cheapest" if aggregate == "cheapest" else "most expensive"
        price = _display_money(
            listing.get("price"), str(listing.get("currency") or "INR"),
            listing.get("purpose") == "rent",
        )
        return {
            "answer": f"The {label} matching property is {listing.get('title')} at {price}.",
            "model": "database",
        }
    if page_info.get("requires_city") and not focused_property:
        purpose = state["intent"].get("purpose")
        action = "rent" if purpose == "rent" else ("buy" if purpose == "buy" else "buy or rent")
        return {
            "answer": f"Which city are you looking to {action} property in?",
            "model": state.get("model", settings.openrouter_model),
        }
    if state["intent"].get("intent") == "property_search" and not focused_property:
        if page_info.get("exhausted"):
            cities = page_info.get("cities") or []
            location = f" in {cities[0]}" if len(cities) == 1 else ""
            if int(page_info.get("offset") or 0) == 0:
                if len(cities) == 1:
                    return {
                        "answer": await _no_inventory_answer(str(cities[0])),
                        "model": "database",
                    }
                return {
                    "answer": f"I couldn’t find a property matching those filters{location}. Would you like to change the budget or requirements?",
                    "model": "database",
                }
            return {
                "answer": "There are no more matching properties. Would you like to change the city or budget?",
                "model": state.get("model", settings.openrouter_model),
            }
        if state["matches"]:
            count = len(state["matches"])
            total = int(page_info.get("total") or count)
            cities = page_info.get("cities") or []
            language = str(state["intent"].get("user_language") or "").lower()
            if page_info.get("nearby_fallback"):
                requested = str(page_info.get("requested_city") or "that city")
                city_names = " and ".join(cities)
                answer = (
                    f"We currently don't have any properties available in {requested}. "
                    f"Here are {count} nearby options from {city_names}. "
                    "If one interests you, tell me its name."
                )
            elif len(cities) >= 2:
                shown = page_info.get("per_city_shown") or {}
                breakdown = " and ".join(
                    f"{shown.get(city, 0)} from {city}" for city in cities
                )
                answer = f"Showing {breakdown}. Reply NEXT to see more from both cities."
            elif language in {"hindi", "hinglish", "hi"}:
                answer = (
                    f"{total} matching properties mili. Abhi {count} dikha raha hoon; "
                    "aur dekhne ke liye NEXT reply karein."
                )
            else:
                answer = (
                    f"Showing {count} of {total} matching properties. "
                    "Reply NEXT to see more, or tell me which one interests you."
                )
            return {
                "answer": answer,
                "model": (
                    "database" if page_info.get("nearby_fallback")
                    else state.get("model", settings.openrouter_model)
                ),
            }
    if state["matches"] and state["intent"].get("intent") == "property_search" and not focused_property:
        count = len(state["matches"])
        language = str(state["intent"].get("user_language") or "").lower()
        if language in {"hindi", "hinglish", "hi"}:
            answer = (
                f"Maine {count} matching properties dikhayi hain. Har image ke neeche "
                "usi property ki details hain—pasand aaye to “I’m interested” tap karein."
            )
        else:
            answer = (
                f"I found {count} matching properties. Each image is paired with that "
                "property’s details. Tell me if a particular property catches your eye."
            )
        return {"answer": answer, "model": state.get("model", settings.openrouter_model)}

    # Recognizable database and knowledge-base questions do not need an external
    # model round-trip. This keeps factual replies fast and quota-independent.
    if focused_property and state.get("matches"):
        return {
            "answer": _offline_answer(state), "model": "database-fallback",
            "grounded": True, "evidence": [{"type": "property", "title": item.get("title")}
                                              for item in state["matches"]],
        }
    if state.get("knowledge"):
        return {
            "answer": _offline_answer(state), "model": "knowledge-base",
            "grounded": True,
            "evidence": [{
                "type": "knowledge", "title": item.get("title"),
                "source": item.get("source"), "score": item.get("retrieval_score"),
                "method": item.get("retrieval_method"),
            } for item in state["knowledge"]],
        }
    if state.get("model") == "deterministic" and state.get("matches"):
        return {"answer": _offline_answer(state), "model": "database", "grounded": True}

    if settings.strict_grounding:
        return {"answer": _offline_answer(state), "model": "grounding-guard", "grounded": False}

    # MongoDB records can contain native datetimes and other BSON-friendly
    # values; stringify those safely when placing inventory in the LLM context.
    inventory = json.dumps(state["matches"], ensure_ascii=False, default=str)
    knowledge = json.dumps(state.get("knowledge", []), ensure_ascii=False, default=str)
    system = """You are Aira, an exceptionally capable, warm real-estate advisor for India.
Be concise and practical. Never invent listings, prices, legal facts, or availability.
When inventory is supplied, base recommendations only on it and explain the fit.
Treat every PROPERTY DATABASE field literally: price/currency/purpose, area_sqft and
carpet_area_sqft, bedrooms/bathrooms/balconies, floor/total_floors, furnishing, facing,
parking_spaces, maintenance_monthly, possession_status, available_from, age_years,
price_negotiable, amenities, nearby, rera_registered, verified, agent and coordinates.
Mention a field only when it exists in the supplied row. A false boolean is a real "No";
a missing field means "not recorded" and must never be guessed.
Treat the supplied KNOWLEDGE BASE as the authoritative source for business facts, policies,
processes and FAQs. Do not invent an answer when the database does not contain the fact;
say that it needs confirmation and ask one concise clarifying question when useful.
When conversation history contains PUBLIC PROPERTY FORM context, use its purpose, city,
budget and reason. Respond like a consultative salesperson: address what the customer said
and ask only the most useful missing question instead of restarting or repeating the form.
Never output a numbered property list or repeat listing fields that the UI renders.
Do not use Markdown bold markers. Keep any text around visual results to 2 sentences.
Ask at most one useful follow-up question. Match the user's language (including Hinglish).
For an uploaded property image, describe visible features, condition clues and design ideas,
but clearly say an image cannot prove structural safety, title, exact value, or location.
Do not output markdown image syntax: the UI renders listing image cards separately.
All financial/legal guidance is general and should be verified professionally."""
    context = (
        f"Detected intent: {json.dumps(state['intent'])}\n"
        f"PROPERTY DATABASE MATCHES: {inventory}\n"
        f"KNOWLEDGE BASE: {knowledge}"
    )
    content: list[dict] = [{"type": "text", "text": f"{state['message']}\n\n{context}"}]
    if state.get("image"):
        content.append({"type": "image_url", "image_url": {"url": state["image"]}})
    prior = [{"role": x["role"], "content": x["content"]} for x in state["history"][-10:]]
    try:
        result = await _create_completion(
            model=settings.openrouter_model,
            messages=[{"role": "system", "content": system}, *prior, {"role": "user", "content": content}],
            temperature=0.35,
            max_tokens=900,
        )
        return {
            "answer": result.choices[0].message.content or _offline_answer(state),
            "answer_ready": True,
            "model": result.model, "grounded": False,
        }
    except Exception:
        return {"answer": _offline_answer(state), "model": "database-fallback"}


builder = StateGraph(AgentState)
builder.add_node("memory", load_context)
builder.add_node("understand", understand)
builder.add_node("inventory", search_inventory)
builder.add_node("property_context", resolve_property_context)
builder.add_node("nearby_transit", enrich_nearby_transit)
builder.add_node("knowledge_retrieval", retrieve_knowledge)
builder.add_node("advisor", compose_answer)
builder.add_edge(START, "memory")
builder.add_edge("memory", "understand")
builder.add_edge("understand", "inventory")
builder.add_edge("inventory", "property_context")
builder.add_edge("property_context", "nearby_transit")
builder.add_edge("nearby_transit", "knowledge_retrieval")
builder.add_edge("knowledge_retrieval", "advisor")
builder.add_edge("advisor", END)
graph = builder.compile()


async def generate_sales_followup(session_id: str) -> str:
    """Write one contextual follow-up while the customer-service window is open."""
    history = await get_history(session_id, settings.max_history_messages)
    prior = [
        {"role": item["role"], "content": item["content"]}
        for item in history[-12:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    prompt = """You are Aira, a concise real-estate sales consultant in India.
Write one natural WhatsApp follow-up based only on the conversation. Give a useful,
specific next step and ask at most one short question. Do not claim a property is
available unless the history confirms it. Do not mention automation, follow-up
schedules, or internal systems. Use the customer's language. Maximum 220 characters."""
    result = await _create_completion(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": prompt},
            *prior,
            {"role": "user", "content": "Send the next helpful sales follow-up now."},
        ],
        temperature=0.35,
        max_tokens=120,
    )
    return (result.choices[0].message.content or "").strip()[:500]
