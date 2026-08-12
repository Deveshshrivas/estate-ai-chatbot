import asyncio
import hmac
import html
import json
import re
import uuid
from datetime import datetime, timezone
from urllib.parse import parse_qs
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import httpx
from bson import ObjectId
from bson.errors import InvalidId
from .agent import (
    advise_on_selected_property, classify_booking_cancellation,
    classify_property_response, graph,
)
from .config import get_settings
from .database import (
    client, db, initialize_database, save_message, seed_demo_properties,
    seed_knowledge_base,
)
from .whatsapp import (
    capture_engaged_lead, extract_messages, is_configured, process_message,
    verify_signature,
)
from .site_visits import display_slot, parse_requested_slot
from . import wacrm


async def capture_web_lead(session_id: str, text: str) -> str | None:
    pending = await db.pending_leads.find_one({"session_id": session_id})
    if not pending:
        booking = await db.leads.find_one(
            {
                "session_id": session_id,
                "$or": [
                    {"appointment_status": {"$in": ["requested", "scheduled", "confirmed"]}},
                    {"status": {"$in": ["booked", "scheduled", "confirmed"]}},
                ],
            },
            sort=[("updated_at", -1)],
        )
        if booking and await classify_booking_cancellation(text, booking):
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            await db.leads.update_one(
                {"_id": booking["_id"]},
                {"$set": {
                    "status": "cancelled", "cancelled_at": now,
                    "appointment_status": "cancelled",
                    "lead_stage": "appointment_cancelled",
                    "cancellation_message": text, "updated_at": now,
                }},
            )
            title = booking.get("property_title") or "your selected property"
            return f"Your consultation request for {title} has been cancelled. You can book again anytime."
        return None
    lowered = text.lower()
    properties = pending.get("matched_properties", [])
    stage = pending.get("stage", "showing_results")
    if stage in {"awaiting_details", "awaiting_booking_name"}:
        customer_name = text.strip().strip("|,.")
        if len(customer_name) < 2 or "@" in customer_name or re.search(r"\d{7,}", customer_name):
            return "What name should I use for the booking?"
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {"stage": "awaiting_booking_email", "customer_name": customer_name}},
        )
        return f"Thanks, {customer_name}. What email should I use for the confirmation?"
    if stage == "awaiting_booking_email":
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        if not email:
            return "Please share a valid email for the confirmation."
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {"stage": "awaiting_booking_phone", "email": email.group(0)}},
        )
        return "What is the best phone number for our advisor to reach you?"
    if stage == "awaiting_booking_phone":
        phone = re.search(r"(?:\+?\d[\d\s-]{7,}\d)", text)
        if not phone:
            return "Please share a valid phone number."
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {"stage": "awaiting_booking_preference", "contact_phone": phone.group(0)}},
        )
        return "Would you prefer a phone call or a video meeting?"
    if stage == "awaiting_booking_preference":
        preference = "video_call" if re.search(r"\b(video|meet|zoom)\b", lowered) else (
            "phone_call" if re.search(r"\b(phone|voice|call|whatsapp call)\b", lowered) else None
        )
        if not preference:
            return "Would a phone call or a video meeting work better for you?"
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {"stage": "awaiting_call_schedule", "contact_preference": preference}},
        )
        method = "video meeting" if preference == "video_call" else "phone call"
        return f"Perfect. What date and time works for the {method}?"
    if stage == "awaiting_call_schedule":
        saved_date = str(pending.get("preferred_date") or "")
        slot, has_date, has_time = parse_requested_slot(text, saved_date)
        if not has_date:
            return "What date works for the consultation?"
        if not has_time:
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": {"preferred_date": text}}
            )
            return "What time works on that date?"
        if not slot or slot <= datetime.now(timezone.utc):
            return "Please choose a future date and time for the consultation."
        now = datetime.now(timezone.utc)
        await db.leads.update_one(
            {"session_id": session_id, "source": "web"},
            {"$set": {
                "name": pending["customer_name"], "email": pending["email"],
                "phone": pending["contact_phone"],
                "contact_preference": pending["contact_preference"], "source": "web",
                "status": "qualified", "lead_stage": "appointment_requested",
                "appointment_status": "requested", "preferred_schedule": slot,
                "property_title": pending.get("selected_property"),
                "matched_properties": properties, "session_id": session_id,
                "updated_at": now,
            }, "$setOnInsert": {"created_at": now}, "$inc": {"enquiry_count": 1}},
            upsert=True,
        )
        await db.pending_leads.delete_one({"_id": pending["_id"]})
        method = "video call" if pending["contact_preference"] == "video_call" else "phone call"
        return (
            f"Thank you, {pending['customer_name']}! Your {method} request for "
            f"{display_slot(slot)} is saved. Our property advisor will confirm it shortly."
        )
    if stage in {"showing_results", "awaiting_property"}:
        try:
            decision = await classify_property_response(text, properties)
        except Exception:
            selected = next(
                (title for title in properties if title.casefold() in text.casefold()),
                None,
            )
            decision = {"is_interested": bool(selected), "selected_property": selected}
        if not decision["is_interested"] and stage == "showing_results":
            return None
        selected = decision.get("selected_property")
        if not selected:
            choices = "\n".join(f"{i + 1}. {title}" for i, title in enumerate(properties[:6]))
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": {"stage": "awaiting_property"}}
            )
            return f"Which property are you interested in? Reply with its name or number:\n{choices}"
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {"stage": "interested", "selected_property": selected}},
        )
        now = datetime.now(timezone.utc)
        await db.leads.update_one(
            {"session_id": session_id, "source": "web"},
            {
                "$set": {
                    "name": "Website visitor",
                    "source": "web",
                    "status": "interested",
                    "lead_stage": "property_selected",
                    "appointment_status": "not_booked",
                    "property_title": selected,
                    "matched_properties": properties,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        return (
            f"Great choice—{selected}. Would you like to book a consultation, "
            "or ask me anything about this property?"
        )

    email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"(?:\+?\d[\d\s-]{7,}\d)", text)
    preference = "video_call" if "video" in lowered else (
        "phone_call" if any(word in lowered for word in ("phone call", "call", "voice")) else None
    )
    if stage in {"interested", "awaiting_details"} and not (email and phone and preference):
        selected = pending.get("selected_property")
        listing = await db.properties.find_one({"title": selected}, {"_id": 0}) or {
            "title": selected
        }
        try:
            decision = await advise_on_selected_property(text, listing)
        except Exception:
            decision = {
                "action": "other",
                "answer": "I can help with details about this property.",
            }
        if decision["action"] == "book":
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": {"stage": "awaiting_booking_name"}}
            )
            return "Great—I can arrange that. What name should I use for the booking?"
        if decision["action"] == "cancel":
            from datetime import datetime, timezone
            await db.pending_leads.update_one(
                {"_id": pending["_id"]},
                {"$set": {
                    "stage": "interested", "booking_cancelled_at": datetime.now(timezone.utc),
                }},
            )
            return (
                f"Your consultation request for {selected} is cancelled. "
                "You can continue asking questions or book again anytime."
            )
        if decision["action"] == "decline":
            return "No problem. Ask me about this property or explore another one."
        answer = decision.get("answer") or "What would you like to know about this property?"
        return f"{answer}\n\nWould you like to book a consultation for this property?"

    return None

settings = get_settings()
static_dir = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await client.admin.command("ping")
    await initialize_database()
    await seed_demo_properties()
    await seed_knowledge_base()
    yield
    await client.close()


app = FastAPI(title="EstateAI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class PropertyIn(BaseModel):
    title: str
    city: str
    locality: str
    type: str
    bedrooms: int = 0
    bathrooms: int = 0
    area_sqft: int
    price: float
    currency: str = "INR"
    purpose: str = Field(pattern="^(buy|rent)$")
    amenities: list[str] = Field(default_factory=list)
    description: str = ""
    images: list[str] = Field(default_factory=list)
    active: bool = True
    featured: bool = False
    verified: bool = False
    project_name: str = ""
    developer: str = ""
    tower_name: str = ""
    unit_number: str = ""
    inventory_status: str = Field(
        default="available",
        pattern="^(available|hold|blocked|booked|sold|archived)$",
    )


class PropertyUpdate(BaseModel):
    title: str | None = None
    city: str | None = None
    locality: str | None = None
    type: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    area_sqft: int | None = None
    price: float | None = None
    currency: str | None = None
    purpose: str | None = Field(default=None, pattern="^(buy|rent)$")
    amenities: list[str] | None = None
    description: str | None = None
    images: list[str] | None = None
    active: bool | None = None
    featured: bool | None = None
    verified: bool | None = None
    project_name: str | None = None
    developer: str | None = None
    tower_name: str | None = None
    unit_number: str | None = None
    inventory_status: str | None = Field(
        default=None,
        pattern="^(available|hold|blocked|booked|sold|archived)$",
    )


class LeadIn(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    phone: str = Field(min_length=7, max_length=20)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    contact_preference: str = Field(pattern="^(phone_call|video_call)$")
    property_title: str
    message: str = ""
    source: str = "web"


class KnowledgeIn(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    content: str = Field(min_length=10, max_length=20_000)
    category: str = Field(default="general", min_length=2, max_length=80)
    keywords: list[str] = Field(default_factory=list)
    source: str = Field(default="business knowledge", max_length=300)
    active: bool = True


@app.get("/")
async def index():
    index_file = static_dir / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"app": "EstateAI", "status": "ok", "wacrm": wacrm.is_configured()}


@app.get("/api/health")
async def health():
    await client.admin.command("ping")
    return {"status": "ok", "database": settings.mongodb_database,
            "model": settings.openrouter_model, "whatsapp": is_configured()}


@app.post("/api/public-property-enquiry", response_class=HTMLResponse)
async def public_property_enquiry(request: Request):
    """Accept a no-JavaScript HTML form and schedule a consented WhatsApp template."""
    raw = (await request.body()).decode("utf-8", errors="replace")
    values = {key: entries[0].strip() for key, entries in parse_qs(raw).items() if entries}
    if values.get("website"):
        return HTMLResponse("<h1>Thank you.</h1>", status_code=200)
    required = ("name", "mobile", "email", "query_reason", "whatsapp_consent")
    if any(not values.get(field) for field in required):
        raise HTTPException(status_code=422, detail="Please complete all required fields and consent")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", values["email"]):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    digits = re.sub(r"\D", "", values["mobile"])
    if len(digits) == 10:
        digits = f"91{digits}"
    if not 10 <= len(digits) <= 15:
        raise HTTPException(status_code=422, detail="Enter a valid WhatsApp mobile number")
    purpose = values.get("property_purpose", "").lower()
    if purpose not in {"buy", "rent", "sell", "invest", ""}:
        raise HTTPException(status_code=422, detail="Invalid property purpose")
    data = {
        "submission_id": str(uuid.uuid4()), "name": values["name"][:100],
        "phone": digits, "email": values["email"][:200],
        "platform_name": values.get("platform_name", "")[:120],
        "property_purpose": purpose, "city": values.get("city", "")[:100],
        "budget": values.get("budget", "")[:100],
        "query_reason": values["query_reason"][:2000],
        "whatsapp_consent": "true",
    }
    result = await wacrm.schedule_public_form_enquiry(data)
    safe_name = html.escape(data["name"])
    scheduled = result["scheduled_for"].strftime("%H:%M UTC")
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enquiry received</title><style>body{{font-family:Arial,sans-serif;background:#f5f7fb;
display:grid;place-items:center;min-height:100vh;margin:0;color:#172033}}main{{background:white;
padding:40px;border-radius:16px;box-shadow:0 12px 40px #17203318;max-width:560px}}
h1{{color:#166534}}p{{line-height:1.6}}a{{color:#4f46e5}}</style></head><body><main>
<h1>Thank you, {safe_name}.</h1><p>Your property enquiry is saved. A personalized WhatsApp message is
scheduled within 2-5 minutes ({scheduled}). Reply to it and Aira will continue
using the details you submitted.</p>
</main></body></html>""")


@app.get("/api/property-image")
async def property_image(url: str = Query(min_length=20, max_length=1000)):
    """Proxy allow-listed listing images with a stable Meta-fetchable response."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "images.unsplash.com":
        raise HTTPException(status_code=400, detail="Unsupported image host")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            upstream = await client.get(url)
            upstream.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Image source unavailable") from exc
    content_type = upstream.headers.get("content-type", "").split(";", 1)[0]
    if not content_type.startswith("image/") or len(upstream.content) > 5_000_000:
        raise HTTPException(status_code=502, detail="Invalid property image")
    return Response(
        content=upstream.content,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400, s-maxage=604800"},
    )


@app.post("/wacrm-webhook")
async def receive_wacrm_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    if not wacrm.verify_signature(body, request.headers.get("X-Wacrm-Signature")):
        raise HTTPException(status_code=401, detail="Invalid WACRM webhook signature")
    payload = json.loads(body)
    event = payload.get("event")
    data = payload.get("data") or {}
    if event == "message.received":
        background_tasks.add_task(wacrm.process_inbound, data)
    elif event == "message.sent" and data.get("content_type") == "template":
        background_tasks.add_task(wacrm.record_template_send, data)
    return {"status": "accepted"}


@app.get("/api/followups/run")
async def run_followups(request: Request):
    authorization = request.headers.get("authorization", "")
    if not settings.cron_secret or authorization != f"Bearer {settings.cron_secret}":
        raise HTTPException(status_code=401, detail="Invalid cron authorization")
    return await wacrm.run_due_followups()


@app.get("/webhooks/whatsapp")
async def verify_whatsapp_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge", "")
    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        return Response(content=challenge, media_type="text/plain")
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhooks/whatsapp")
async def receive_whatsapp_webhook(request: Request):
    body = await request.body()
    if not verify_signature(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    payload = json.loads(body)
    # Meta expects an immediate 200; processing continues outside the response path.
    for message in extract_messages(payload):
        asyncio.create_task(process_message(message))
    return {"status": "accepted"}


def require_property_admin(request: Request) -> None:
    supplied = request.headers.get("x-property-admin-key", "")
    if not settings.property_admin_secret or not hmac.compare_digest(
        supplied, settings.property_admin_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid property admin credentials")


@app.get("/api/admin/properties")
async def admin_list_properties(
    request: Request, search: str = "", city: str = "", purpose: str = "",
    project: str = "", tower: str = "", property_type: str = "",
    inventory_status: str = "", bedrooms: int | None = Query(default=None, ge=0, le=20),
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    status: str = "all", page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
):
    require_property_admin(request)
    query: dict[str, Any] = {}
    if city:
        query["city"] = {"$regex": f"^{re.escape(city)}$", "$options": "i"}
    if purpose in {"buy", "rent"}:
        query["purpose"] = purpose
    for value, field in (
        (project, "project_name"), (tower, "tower_name"),
        (property_type, "type"), (inventory_status, "inventory_status"),
    ):
        if value:
            query[field] = {"$regex": f"^{re.escape(value)}$", "$options": "i"}
    if bedrooms is not None:
        query["bedrooms"] = bedrooms
    price_query: dict[str, float] = {}
    if min_price is not None:
        price_query["$gte"] = min_price
    if max_price is not None:
        price_query["$lte"] = max_price
    if price_query:
        query["price"] = price_query
    if status == "active":
        query["active"] = {"$ne": False}
    elif status == "archived":
        query["active"] = False
    if search:
        escaped = re.escape(search)
        query["$or"] = [
            {field: {"$regex": escaped, "$options": "i"}}
            for field in (
                "title", "project_name", "developer", "tower_name", "unit_number",
                "city", "locality", "type", "description",
            )
        ]
    total = await db.properties.count_documents(query)
    cursor = db.properties.find(query).sort([
        ("featured", -1), ("updated_at", -1), ("created_at", -1), ("title", 1)
    ]).skip((page - 1) * limit).limit(limit)
    items = []
    async for item in cursor:
        item["id"] = str(item.pop("_id"))
        items.append(item)
    projects, towers, property_types, bedroom_options = await asyncio.gather(
        db.properties.distinct("project_name"),
        db.properties.distinct("tower_name"),
        db.properties.distinct("type"),
        db.properties.distinct("bedrooms"),
    )
    available_count = await db.properties.count_documents({
        **query, "inventory_status": {"$in": ["available", None, ""]},
    })
    return jsonable_encoder({
        "items": items, "total": total, "page": page,
        "pages": max(1, (total + limit - 1) // limit),
        "cities": sorted(await db.properties.distinct("city")),
        "projects": sorted(str(value) for value in projects if value),
        "towers": sorted(str(value) for value in towers if value),
        "property_types": sorted(str(value) for value in property_types if value),
        "bedroom_options": sorted(int(value) for value in bedroom_options if isinstance(value, (int, float))),
        "summary": {
            "projects": len([value for value in projects if value]),
            "towers": len([value for value in towers if value]),
            "units": total,
            "available": available_count,
        },
    })


@app.post("/api/admin/properties")
async def admin_add_property(request: Request, item: PropertyIn):
    require_property_admin(request)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    result = await db.properties.insert_one({
        **item.model_dump(), "created_at": now, "updated_at": now,
    })
    return {"id": str(result.inserted_id)}


@app.patch("/api/admin/properties/{property_id}")
async def admin_update_property(property_id: str, request: Request, item: PropertyUpdate):
    require_property_admin(request)
    try:
        object_id = ObjectId(property_id)
    except InvalidId as exc:
        raise HTTPException(status_code=400, detail="Invalid property ID") from exc
    from datetime import datetime, timezone
    changes = item.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail="No property changes supplied")
    changes["updated_at"] = datetime.now(timezone.utc)
    result = await db.properties.update_one({"_id": object_id}, {"$set": changes})
    if not result.matched_count:
        raise HTTPException(status_code=404, detail="Property not found")
    return {"success": True}


@app.delete("/api/admin/properties/{property_id}")
async def admin_archive_property(property_id: str, request: Request):
    require_property_admin(request)
    try:
        object_id = ObjectId(property_id)
    except InvalidId as exc:
        raise HTTPException(status_code=400, detail="Invalid property ID") from exc
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    result = await db.properties.update_one(
        {"_id": object_id},
        {"$set": {"active": False, "archived_at": now, "updated_at": now}},
    )
    if not result.matched_count:
        raise HTTPException(status_code=404, detail="Property not found")
    return {"success": True, "recoverable": True}


@app.post("/api/knowledge")
async def add_knowledge(item: KnowledgeIn):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    record = {**item.model_dump(), "created_at": now, "updated_at": now}
    result = await db.knowledge_base.insert_one(record)
    return {"id": str(result.inserted_id)}


@app.get("/api/knowledge")
async def list_knowledge(
    category: str = "", page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    query = {"category": category} if category else {}
    total = await db.knowledge_base.count_documents(query)
    cursor = db.knowledge_base.find(query, {"_id": 0}).sort("updated_at", -1).skip(
        (page - 1) * limit
    ).limit(limit)
    items = [item async for item in cursor]
    return jsonable_encoder({"items": items, "total": total, "page": page})


@app.post("/api/leads")
async def create_lead(lead: LeadIn):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    result = await db.leads.update_one(
        {"phone": lead.phone, "property_title": lead.property_title},
        {
            "$set": {**lead.model_dump(), "status": "new", "updated_at": now},
            "$setOnInsert": {"created_at": now},
            "$inc": {"enquiry_count": 1},
        },
        upsert=True,
    )
    return {"success": True, "lead_id": str(result.upserted_id or "updated")}


@app.get("/api/properties")
async def list_properties(
    search: str = "",
    city: str = "",
    purpose: str = "",
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=12, ge=1, le=50),
):
    query: dict = {"active": {"$ne": False}}
    if city:
        query["city"] = {"$regex": f"^{city}$", "$options": "i"}
    if purpose in {"buy", "rent"}:
        query["purpose"] = purpose
    if search:
        escaped = __import__("re").escape(search)
        query["$or"] = [
            {"title": {"$regex": escaped, "$options": "i"}},
            {"city": {"$regex": escaped, "$options": "i"}},
            {"locality": {"$regex": escaped, "$options": "i"}},
            {"type": {"$regex": escaped, "$options": "i"}},
        ]
    total = await db.properties.count_documents(query)
    cursor = (
        db.properties.find(query, {"_id": 0})
        .sort([("featured", -1), ("created_at", -1), ("title", 1)])
        .skip((page - 1) * limit)
        .limit(limit)
    )
    items = [item async for item in cursor]
    cities = await db.properties.distinct("city")
    return jsonable_encoder({
        "items": items,
        "total": total,
        "page": page,
        "pages": max(1, (total + limit - 1) // limit),
        "cities": sorted(cities),
    })


@app.websocket("/ws/chat/{session_id}")
async def chat_socket(ws: WebSocket, session_id: str):
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            message_id = str(payload.get("id", ""))
            text = str(payload.get("message", "")).strip()
            image = payload.get("image")
            if not text and not image:
                continue
            await ws.send_json({"type": "status", "id": message_id, "status": "delivered"})
            await asyncio.sleep(.25)
            await ws.send_json({"type": "status", "id": message_id, "status": "read"})
            await ws.send_json({"type": "typing", "active": True})
            await save_message(session_id, "user", text or "Please analyze this property image.",
                               has_image=bool(image))
            try:
                lead_reply = await capture_web_lead(session_id, text)
                if lead_reply:
                    await save_message(session_id, "assistant", lead_reply)
                    await ws.send_json({"type": "message", "content": lead_reply,
                                        "listings": [], "model": "lead-capture", "id": message_id})
                    continue
                result = await graph.ainvoke({
                    "session_id": session_id,
                    "message": text or "Please analyze this property image.",
                    "image": image,
                })
                listings = result.get("matches", [])
                intent = result.get("intent", {})
                answer = result["answer"]
                await capture_engaged_lead(
                    session_id=session_id,
                    source="web",
                    intent=intent,
                    matches=listings,
                    last_message=text,
                )
                if listings:
                    await db.pending_leads.update_one(
                        {"session_id": session_id},
                        {"$set": {
                            "session_id": session_id,
                            "matched_properties": [x.get("title") for x in listings],
                            "preferences": intent,
                            "stage": "showing_results",
                            "selected_property": None,
                        }},
                        upsert=True,
                    )
                await save_message(session_id, "assistant", answer,
                                   listings=[x.get("title") for x in listings])
                await ws.send_json({
                    "type": "message", "content": answer,
                    "listings": jsonable_encoder(listings),
                    "model": result.get("model"), "id": message_id,
                    "grounded": result.get("grounded"),
                    "evidence": jsonable_encoder(result.get("evidence") or []),
                })
            except Exception as exc:
                await ws.send_json({
                    "type": "error",
                    "message": "I couldn't reach the AI service. Check your API key, free-model rate limit, and MongoDB connection.",
                    "detail": str(exc) if app.debug else None,
                })
            finally:
                await ws.send_json({"type": "typing", "active": False})
    except WebSocketDisconnect:
        return
