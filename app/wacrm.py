"""WACRM transport and outbound-template follow-up orchestration for EstateAI."""

import asyncio
import hashlib
import hmac
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from .agent import (
    finalize_agent_answer, generate_advisor_reply, generate_sales_followup, graph,
)
from .config import get_settings
from .database import db, save_message
from .whatsapp import (
    _cancel_saved_booking, _format_price, _handle_customer_identity, _handle_lead_details,
    capture_engaged_lead, delivery_image_url, mark_read_and_typing,
)

settings = get_settings()
SIGNATURE_TOLERANCE_SECONDS = 300

OPT_OUT_PHRASES = (
    "not interested", "not intrested", "no interest", "stop", "unsubscribe",
    "don't contact", "do not contact", "dont contact", "remove me", "no thanks",
)

PROPERTY_OPT_OUT_PHRASES = (
    "don't want property", "do not want property", "dont want property",
    "not looking for property", "no longer looking", "stop property search",
    "close my enquiry", "close my inquiry",
)


def is_configured() -> bool:
    return bool(settings.wacrm_api_url and settings.wacrm_api_key)


def verify_signature(body: bytes, signature: str | None) -> bool:
    if not settings.wacrm_webhook_secret or not signature:
        return False
    try:
        parts = dict(part.split("=", 1) for part in signature.split(","))
        timestamp = parts["t"]
        supplied = parts["v1"]
        if abs(time.time() - int(timestamp)) > SIGNATURE_TOLERANCE_SECONDS:
            return False
    except (KeyError, ValueError):
        return False
    expected = hmac.new(
        settings.wacrm_webhook_secret.encode(),
        f"{timestamp}.".encode() + body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, supplied)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.wacrm_api_key}"}


async def _request(
    method: str, path: str, *, retryable: bool = False, **kwargs
) -> dict[str, Any] | None:
    if not is_configured():
        return None
    attempts = 3 if retryable else 1
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
        for attempt in range(attempts):
            try:
                response = await client.request(
                    method, f"{settings.wacrm_api_url.rstrip('/')}{path}",
                    headers=_headers(), **kwargs,
                )
                response.raise_for_status()
                payload = response.json()
                return payload.get("data", payload)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt + 1 >= attempts:
                    raise
            except httpx.HTTPStatusError as exc:
                if attempt + 1 >= attempts or exc.response.status_code not in {
                    429, 500, 502, 503, 504,
                }:
                    raise
            await asyncio.sleep(0.25 * (2 ** attempt))
    return None


async def get_contact(contact_id: str) -> dict[str, Any] | None:
    if not contact_id:
        return None
    return await _request("GET", f"/api/v1/contacts/{contact_id}", retryable=True)


async def mark_read(message_id: str) -> None:
    """Best-effort read receipt and typing indicator; never block a reply."""
    if not message_id:
        return
    try:
        # Some WACRM deployments expose this endpoint. Prefer it when present
        # so message transport stays centralized.
        await _request(
            "POST", "/api/v1/messages/read",
            retryable=True, json={"message_id": message_id},
        )
        return
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            # A receipt is optional. Continue to the direct Meta fallback.
            pass
    except Exception:
        pass
    try:
        # Current WACRM public API has no read endpoint. The chatbot already
        # has the same Meta phone configuration, so use Cloud API directly.
        await mark_read_and_typing(message_id)
    except Exception:
        # Invalid/expired receipt credentials must not suppress the AI reply.
        pass


async def send_text(phone: str, text: str) -> dict[str, Any] | None:
    return await _request(
        "POST", "/api/v1/messages",
        json={"to": phone if phone.startswith("+") else f"+{phone}", "type": "text", "text": text[:4096]},
    )


async def send_image(phone: str, image_url: str, caption: str) -> dict[str, Any] | None:
    return await _request(
        "POST", "/api/v1/messages",
        json={
            "to": phone if phone.startswith("+") else f"+{phone}",
            "type": "image", "media_url": image_url, "text": caption[:1024],
        },
    )


async def send_template(
    phone: str, name: str, language: str, params: list[str] | None = None
) -> dict[str, Any] | None:
    return await _request(
        "POST", "/api/v1/messages",
        json={
            "to": phone if phone.startswith("+") else f"+{phone}",
            "type": "template",
            "template": {"name": name, "language": language, "params": params or []},
        },
    )


def _form_budget_inr(value: str) -> float | None:
    """Parse a concise INR form budget for CRM/pipeline value."""
    match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(crore|cr|lakh|lac|k|thousand)?",
        value.replace(",", "").casefold(),
    )
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2) or ""
    multiplier = 10_000_000 if unit in {"crore", "cr"} else (
        100_000 if unit in {"lakh", "lac"} else (
            1_000 if unit in {"k", "thousand"} else 1
        )
    )
    return amount * multiplier


async def schedule_public_form_enquiry(data: dict[str, str]) -> dict[str, Any]:
    """Persist consented form context and schedule its first approved template."""
    now = datetime.now(timezone.utc)
    phone = data["phone"]
    contact = await _request(
        "POST", "/api/v1/contacts",
        json={
            "phone": f"+{phone}", "name": data["name"], "email": data["email"],
            "company": data.get("platform_name") or None,
            "tags": ["public-property-form"],
        },
    )
    if not contact:
        raise RuntimeError("WACRM contact could not be created")
    # Stable per submission, so retries keep the same deadline while new
    # enquiries are naturally spread across the requested 2-5 minute window.
    delay_minutes = 2 + (
        int(hashlib.sha256(data["submission_id"].encode()).hexdigest()[:2], 16) % 4
    )
    due_at = now + timedelta(minutes=delay_minutes)
    form_record = {
        **data, "contact_id": contact.get("id"), "status": "template_scheduled",
        "template_due_at": due_at, "created_at": now, "updated_at": now,
    }
    await db.public_form_leads.insert_one(form_record)
    budget_value = _form_budget_inr(data.get("budget") or "")
    await db.leads.update_one(
        {"whatsapp_phone": phone},
        {
            "$set": {
                "name": data["name"], "phone": phone, "whatsapp_phone": phone,
                "email": data["email"], "source": "public_property_form",
                "form_submission_id": data["submission_id"],
                "wacrm_contact_id": contact.get("id"),
                "property_purpose": data.get("property_purpose") or None,
                "city": data.get("city") or None,
                "budget_max": budget_value,
                "budget_currency": "INR", "query_reason": data["query_reason"],
                "platform_name": data.get("platform_name") or None,
                "status": "new", "lead_stage": "form_submitted",
                "appointment_status": "not_booked", "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    context = (
        "PUBLIC PROPERTY FORM (customer-provided context): "
        f"Name={data['name']}; Purpose={data.get('property_purpose') or 'not specified'}; "
        f"City={data.get('city') or 'not specified'}; Budget={data.get('budget') or 'not specified'}; "
        f"Platform={data.get('platform_name') or 'not specified'}; "
        f"Reason={data['query_reason']}. Use this context to ask relevant consultative sales questions."
    )
    await save_message(f"wa:{phone}", "system", context, channel="public_form")
    await db.outbound_campaigns.update_many(
        {"phone": phone, "status": "active"},
        {"$set": {"status": "superseded", "next_followup_at": None, "updated_at": now}},
    )
    await db.outbound_campaigns.insert_one({
        "phone": phone, "contact_id": contact.get("id"),
        "source": "public_property_form", "submission_id": data["submission_id"],
        "initial_template_name": settings.public_form_template_name,
        "template_language": settings.public_form_template_language,
        "template_params": [
            data["name"][:60],
            (data.get("property_purpose") or "property")[:60],
            (data.get("city") or "your preferred city")[:60],
            data["query_reason"][:120],
        ],
        "status": "active", "attempts": 0, "started_at": now,
        "updated_at": now, "last_customer_message_at": None,
        "next_followup_at": due_at,
    })
    # Create the pipeline opportunity immediately. A WACRM permission or
    # network error is recorded on the Mongo lead and never loses the form.
    await _sync_new_lead(phone, str(contact.get("id") or ""), None)
    return {
        "submission_id": data["submission_id"], "scheduled_for": due_at,
        "delay_minutes": delay_minutes,
    }


async def record_template_send(data: dict[str, Any]) -> None:
    """Start follow-up state only for an actual outbound template event."""
    message_id = str(data.get("whatsapp_message_id") or "")
    if not message_id:
        return
    try:
        await db.webhook_events.insert_one({
            "message_id": f"sent:{message_id}", "created_at": datetime.now(timezone.utc)
        })
    except DuplicateKeyError:
        return
    contact = await get_contact(str(data.get("contact_id") or ""))
    if not contact or not contact.get("phone"):
        return
    phone = str(contact["phone"]).lstrip("+")
    now = datetime.now(timezone.utc)
    existing = await db.outbound_campaigns.find_one({"phone": phone, "status": "active"})
    if existing:
        # A scheduled follow-up is itself a template send. Record it without
        # resetting the attempt counter or extending its already-computed due time.
        await db.outbound_campaigns.update_one(
            {"_id": existing["_id"]},
            {"$set": {"last_outbound_message_id": message_id, "updated_at": now}},
        )
        return
    await db.outbound_campaigns.insert_one({
        "phone": phone,
        "contact_id": data.get("contact_id"),
        "conversation_id": data.get("conversation_id"),
        "initial_template_name": data.get("template_name"),
        "template_language": data.get("template_language") or "en_US",
        "whatsapp_message_id": message_id,
        "status": "active",
        "attempts": 0,
        "started_at": now,
        "updated_at": now,
        "last_customer_message_at": None,
        "next_followup_at": now + timedelta(hours=settings.followup_delay_hours),
    })


def _is_opt_out(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(phrase in normalized for phrase in OPT_OUT_PHRASES)


def _is_property_opt_out(text: str) -> bool:
    normalized = " ".join(text.lower().replace("’", "'").split())
    # Rejecting a city/listing while asking for another one is a preference
    # correction, not permission to close the lead or stop all messages.
    if any(phrase in normalized for phrase in PROPERTY_OPT_OUT_PHRASES):
        return True
    explicit_stop = (
        "unsubscribe", "don't contact", "do not contact",
        "dont contact", "remove me",
    )
    if any(phrase in normalized for phrase in explicit_stop):
        return True
    if re.search(r"\bstop\b.*\b(messages?|follow-?ups?|contacting)\b", normalized):
        return True
    has_new_preference = bool(re.search(
        r"\b(but|instead|want|looking|show|prefer|need|another|other)\b",
        normalized,
    ))
    if has_new_preference:
        return False
    short_reply = normalized.strip(" .!?,")
    return short_reply in {
        "not interested", "not intrested", "no interest", "no thanks",
        "no thanks not interested", "no thanks i am not interested",
        "stop", "please stop",
    }


def _might_be_conversation_control(text: str) -> bool:
    """Cheap routing only; the LLM makes the actual semantic decision."""
    normalized = " ".join(text.lower().replace("’", "'").split())
    return any(
        phrase in normalized
        for phrase in (*OPT_OUT_PHRASES, *PROPERTY_OPT_OUT_PHRASES)
    )


def _is_goodbye(text: str) -> bool:
    normalized = " ".join(text.lower().split()).strip(" .!?,")
    return normalized in {"bye", "goodbye", "thanks bye", "thank you bye", "ok bye"}


def _asks_available_cities(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    has_question = bool(re.search(r"\b(what|which|wich|wgich|where|list|available)\b", normalized))
    has_location = bool(re.search(r"\b(cit(?:y|ies)|locations?|markets?)\b", normalized))
    has_inventory = bool(re.search(r"\b(propert(?:y|ies)|have|available|operate)\b", normalized))
    return has_question and has_location and has_inventory


def _asks_for_call(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return bool(re.search(
        r"\b(can|could|may|want|need|like|prefer|schedule|book|arrange)\b.*"
        r"\b(call|callback|phone call|voice call|video call|meeting)\b",
        normalized,
    ))


async def _register_turn(phone: str, message_id: str) -> None:
    await db.conversation_turns.update_one(
        {"_id": phone},
        {"$set": {
            "latest_message_id": message_id,
            "received_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )


async def _is_current_turn(phone: str, message_id: str) -> bool:
    turn = await db.conversation_turns.find_one({"_id": phone})
    return bool(turn and turn.get("latest_message_id") == message_id)


async def _acquire_conversation_lock(phone: str) -> str | None:
    """Serialize one customer's turns across workers and serverless instances."""
    token = uuid.uuid4().hex
    for _ in range(100):
        now = datetime.now(timezone.utc)
        try:
            await db.conversation_locks.insert_one({
                "_id": phone,
                "token": token,
                "locked_until": now + timedelta(seconds=90),
            })
            return token
        except DuplicateKeyError:
            await db.conversation_locks.delete_one({
                "_id": phone, "locked_until": {"$lte": now}
            })
            await asyncio.sleep(0.2)
    return None


async def _release_conversation_lock(phone: str, token: str) -> None:
    await db.conversation_locks.delete_one({"_id": phone, "token": token})


async def _sync_new_lead(phone: str, contact_id: str, conversation_id: str | None) -> None:
    lead = await db.leads.find_one(
        {"$or": [{"whatsapp_phone": phone}, {"phone": phone}]},
        sort=[("updated_at", -1)],
    )
    if not lead:
        return
    notes = [
        "EstateAI property lead",
        f"Stage: {lead.get('lead_stage') or lead.get('status') or 'discovery'}",
        f"Property: {lead.get('property_title') or 'Not selected'}",
        f"City: {lead.get('city') or ', '.join(lead.get('cities') or []) or 'Not provided'}",
        f"Schedule: {lead.get('preferred_schedule') or 'Not provided'}",
        f"Booking type: {lead.get('booking_type') or 'Not selected'}",
        f"Preference: {lead.get('contact_preference') or 'Not provided'}",
        f"Email: {lead.get('email') or 'Not provided'}",
        f"WhatsApp: +{phone}",
    ]
    value = lead.get("budget_max")
    lead_stage = str(lead.get("lead_stage") or lead.get("status") or "new")
    pipeline_stage = (
        "Visit Planned" if lead_stage in {"site_visit_scheduled", "visit_planned"}
        else "Won" if lead_stage in {"booked", "won"}
        else "Lost" if lead_stage in {"closed", "cancelled", "appointment_cancelled"}
        else "Qualified" if lead_stage in {
            "property_selected", "qualified", "appointment_requested", "call_requested"
        }
        else "New"
    )
    payload = {
        "contact_id": contact_id or lead.get("wacrm_contact_id"),
        "conversation_id": conversation_id,
        "title": f"{lead.get('name') or phone} - Property enquiry",
        "notes": "\n".join(notes), "stage": pipeline_stage,
        **({"value": float(value), "currency": "INR"}
           if isinstance(value, (int, float)) and value >= 0 else {}),
    }
    try:
        deal_id = str(lead.get("wacrm_deal_id") or "")
        result = await _request(
            "PATCH" if deal_id else "POST",
            f"/api/v1/deals/{deal_id}" if deal_id else "/api/v1/deals",
            json=payload,
        )
    except Exception as exc:
        # MongoDB remains the authoritative lead store. WACRM deal sync is a
        # secondary side effect and must never suppress the customer reply.
        await db.leads.update_one(
            {"_id": lead["_id"]},
            {"$set": {
                "wacrm_sync_pending": True,
                "wacrm_sync_error": str(exc)[:500],
                "wacrm_sync_failed_at": datetime.now(timezone.utc),
            }},
        )
        return
    if result:
        await db.leads.update_one(
            {"_id": lead["_id"]},
            {
                "$set": {
                    "wacrm_synced_at": datetime.now(timezone.utc),
                    "wacrm_deal_id": result.get("id") or lead.get("wacrm_deal_id"),
                    "wacrm_sync_pending": False,
                },
                "$unset": {"wacrm_sync_error": "", "wacrm_sync_failed_at": ""},
            },
        )


async def _sync_cancelled_lead(
    phone: str, contact_id: str, conversation_id: str | None
) -> None:
    """Mirror an AI-confirmed MongoDB cancellation into the WACRM deal."""
    lead = await db.leads.find_one(
        {
            "$or": [{"whatsapp_phone": phone}, {"phone": phone}],
            "status": "cancelled",
        },
        sort=[("updated_at", -1)],
    )
    if not lead or lead.get("wacrm_cancelled_synced_at"):
        return
    cancelled_at = lead.get("cancelled_at")
    cancelled_text = str(lead.get("cancellation_message") or "Customer requested cancellation")
    notes = [
        "EstateAI consultation cancelled by customer",
        f"Property: {lead.get('property_title') or 'Not selected'}",
        f"Cancelled at: {cancelled_at}",
        f"Customer message: {cancelled_text}",
        f"WhatsApp: +{phone}",
    ]
    try:
        deal_id = str(lead.get("wacrm_deal_id") or "")
        result = await _request(
            "PATCH" if deal_id else "POST",
            f"/api/v1/deals/{deal_id}" if deal_id else "/api/v1/deals",
            json={
                "contact_id": contact_id,
                "conversation_id": conversation_id,
                "title": f"{lead.get('name') or phone} - Cancelled consultation",
                "notes": "\n".join(notes),
                "stage": "Lost",
                "status": "lost",
            },
        )
    except Exception as exc:
        await db.leads.update_one(
            {"_id": lead["_id"]},
            {"$set": {
                "wacrm_cancel_sync_pending": True,
                "wacrm_cancel_sync_error": str(exc)[:500],
                "wacrm_cancel_sync_failed_at": datetime.now(timezone.utc),
            }},
        )
        return
    if result:
        await db.leads.update_one(
            {"_id": lead["_id"]},
            {
                "$set": {
                    "wacrm_cancelled_synced_at": datetime.now(timezone.utc),
                    "wacrm_deal_id": result.get("id") or lead.get("wacrm_deal_id"),
                    "wacrm_cancel_sync_pending": False,
                },
                "$unset": {"wacrm_cancel_sync_error": "", "wacrm_cancel_sync_failed_at": ""},
            },
        )


async def process_inbound(data: dict[str, Any]) -> None:
    """Deduplicate and serialize genuine WACRM inbound messages per customer."""
    message_id = str(data.get("whatsapp_message_id") or "")
    if not message_id:
        return
    try:
        await db.webhook_events.insert_one({
            "message_id": message_id, "created_at": datetime.now(timezone.utc)
        })
    except DuplicateKeyError:
        return
    try:
        contact = await get_contact(str(data.get("contact_id") or ""))
        if not contact or not contact.get("phone"):
            return
        phone = str(contact["phone"]).lstrip("+")
        text = str(data.get("text") or "").strip()
        if not text:
            return
        await _register_turn(phone, message_id)
        token = await _acquire_conversation_lock(phone)
        if not token:
            await db.webhook_errors.insert_one({
                "message_id": message_id,
                "error": "Timed out waiting for the per-customer conversation lock",
                "created_at": datetime.now(timezone.utc),
            })
            raise RuntimeError("Timed out waiting for the conversation lock")
        try:
            if not await _is_current_turn(phone, message_id):
                return
            await _process_current_turn(data, message_id, contact, phone, text)
        finally:
            await _release_conversation_lock(phone, token)
    except Exception:
        # A failed event must be claimable when WACRM retries the webhook.
        # Keeping the unique id after failure would silently discard that retry.
        await db.webhook_events.delete_one({"message_id": message_id})
        raise


async def _process_current_turn(
    data: dict[str, Any], message_id: str, contact: dict[str, Any], phone: str, text: str
) -> None:
    """Process one turn while its per-customer lock is held."""
    now = datetime.now(timezone.utc)

    async def send_current_text(to: str, body: str) -> Any:
        if await _is_current_turn(phone, message_id):
            return await send_text(to, body)
        return None

    async def send_current_image(to: str, url: str, caption: str) -> Any:
        if await _is_current_turn(phone, message_id):
            return await send_image(to, url, caption)
        return None

    async def write_reply(required_message: str, context: dict[str, Any] | None = None) -> str:
        return await generate_advisor_reply(
            required_message, text, context or {},
        )
    await db.public_form_leads.update_many(
        {"phone": phone, "status": {"$in": ["template_scheduled", "template_sent"]}},
        {"$set": {"status": "customer_replied", "replied_at": now, "updated_at": now}},
    )
    campaign = await db.outbound_campaigns.find_one({"phone": phone, "status": "active"})
    should_opt_out = (
        _is_property_opt_out(text)
        if _might_be_conversation_control(text) else False
    )
    if should_opt_out:
        await db.outbound_campaigns.update_many(
            {"phone": phone, "status": "active"},
            {"$set": {"status": "opted_out", "opted_out_at": now, "next_followup_at": None}},
        )
        await db.pending_leads.delete_many({"session_id": f"wa:{phone}"})
        await db.search_sessions.delete_many({"session_id": f"wa:{phone}"})
        await db.leads.update_many(
            {"$or": [{"whatsapp_phone": phone}, {"phone": phone}]},
            {"$set": {
                "status": "closed_not_interested", "lead_stage": "closed",
                "appointment_status": "not_requested", "closed_at": now,
                "last_customer_message": text, "updated_at": now,
            }},
        )
        answer = await write_reply(
            "Understood. We have stopped all sales follow-ups and will not message you again. Thank you.",
            {"campaign_status": "opted_out"},
        )
        await send_current_text(phone, answer)
        return
    if campaign:
        await db.outbound_campaigns.update_one(
            {"_id": campaign["_id"]},
            {"$set": {
                "last_customer_message_at": now,
                "next_followup_at": now + timedelta(hours=settings.followup_delay_hours),
                "updated_at": now,
            }},
        )
    await mark_read(message_id)
    session_id = f"wa:{phone}"
    await save_message(session_id, "user", text, channel="whatsapp")
    if await _handle_customer_identity(phone, text, send=send_current_text):
        return
    if await _cancel_saved_booking(
        phone, text, send=send_current_text, ai_replies=True,
    ):
        await _sync_cancelled_lead(
            phone, str(data.get("contact_id") or ""), data.get("conversation_id")
        )
        return
    if _is_goodbye(text):
        await db.pending_leads.delete_many({"session_id": f"wa:{phone}"})
        if await _is_current_turn(phone, message_id):
            answer = await write_reply(
                "Thanks for chatting with us. You can message again whenever you need property help.",
                {"contact_name": contact.get("name") or ""},
            )
            await send_text(phone, answer)
            await save_message(f"wa:{phone}", "assistant", answer, channel="whatsapp")
        return
    if _asks_available_cities(text):
        cities = sorted(str(city) for city in await db.properties.distinct("city") if city)
        answer = (
            "We currently have properties in " + ", ".join(cities) + ". Which city interests you?"
            if cities else "Our live inventory is being updated. Which city interests you?"
        )
        answer = await write_reply(answer, {"available_cities": cities})
        if await _is_current_turn(phone, message_id):
            await send_text(phone, answer)
            await save_message(f"wa:{phone}", "assistant", answer, channel="whatsapp")
        return
    if _asks_for_call(text):
        await db.leads.update_one(
            {"whatsapp_phone": phone},
            {"$set": {
                "name": contact.get("name") or phone, "phone": phone,
                "whatsapp_phone": phone, "source": "whatsapp",
                "status": "engaged", "lead_stage": "call_requested",
                "appointment_status": "awaiting_schedule",
                "contact_preference": "phone_call",
                "last_customer_message": text, "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        await db.pending_leads.update_one(
            {"session_id": f"wa:{phone}"},
            {"$set": {
                "session_id": f"wa:{phone}", "phone": phone,
                "whatsapp_name": contact.get("name") or "",
                "stage": "awaiting_call_schedule",
                "wacrm_contact_id": data.get("contact_id"),
                "wacrm_conversation_id": data.get("conversation_id"),
                "updated_at": now,
            }},
            upsert=True,
        )
        answer = await write_reply(
            "Yes, I can arrange a callback with a property advisor. What future day and time works for you?",
            {"contact_name": contact.get("name") or ""},
        )
        if await _is_current_turn(phone, message_id):
            await send_text(phone, answer)
            await save_message(f"wa:{phone}", "assistant", answer, channel="whatsapp")
        await _sync_new_lead(
            phone, str(data.get("contact_id") or ""), data.get("conversation_id")
        )
        return
    if await _handle_lead_details(
        phone, str(contact.get("name") or ""), text,
        send=send_current_text, send_property_image=send_current_image,
        ai_replies=True,
    ):
        await _sync_new_lead(phone, str(data.get("contact_id") or ""), data.get("conversation_id"))
        return
    result = await graph.ainvoke({"session_id": session_id, "message": text, "image": None})
    # A newer customer message arrived while the model was thinking. Never send
    # this stale turn's text or property gallery.
    if not await _is_current_turn(phone, message_id):
        return
    matches = result.get("matches", [])
    intent = result.get("intent", {})
    focused_property = bool((result.get("page_info") or {}).get("focused_property"))
    captured = await capture_engaged_lead(
        phone=phone,
        name=str(contact.get("name") or ""),
        intent=intent,
        matches=matches,
        last_message=text,
    )
    if captured:
        await _sync_new_lead(
            phone, str(data.get("contact_id") or ""), data.get("conversation_id")
        )
    # Search results are visual by default. A factual question about one selected
    # property sends media only when the customer actually asks for photos.
    should_send_images = bool(matches) and (
        intent.get("intent") in {"property_search", "compare"}
        or intent.get("wants_images") is True
    )
    # Results are page-capped by the shared agent: five for one city,
    # or three from each requested city.
    listings_to_send = matches if should_send_images else []
    caption_cards = []
    for listing in listings_to_send:
        fallback = (
            f"🏡 {listing['title']}\n📍 {listing['locality']}, {listing['city']}\n"
            f"{listing.get('bedrooms', 0)} BHK · {listing.get('area_sqft', 0):,} sq ft\n"
            f"💰 {_format_price(listing)}\n\n"
            "Like this one? Send its name and I'll share more details."
        )
        caption_cards.append({
            "title": str(listing["title"]),
            "location": f"{listing['locality']}, {listing['city']}",
            "configuration": f"{listing.get('bedrooms', 0)} BHK",
            "area": f"{listing.get('area_sqft', 0):,} sq ft",
            "price": _format_price(listing),
            "fallback": fallback,
        })
    captions = [card["fallback"] for card in caption_cards]
    for listing, caption in zip(listings_to_send, captions):
        images = listing.get("images") or []
        if not images:
            continue
        selected_images = images if focused_property and intent.get("wants_images") else images[:1]
        for image_url in selected_images:
            if not await _is_current_turn(phone, message_id):
                return
            await send_image(phone, delivery_image_url(image_url), caption)
    if not result.get("answer"):
        result["answer"] = "Ask which city, budget and property type the customer prefers."
    answer = await finalize_agent_answer(
        result,
        text,
        {
            "intent": intent,
            "properties": [item.get("title") for item in matches],
        },
    )
    if not await _is_current_turn(phone, message_id):
        return
    await send_text(phone, answer)
    await save_message(session_id, "assistant", answer, channel="whatsapp")
    if matches:
        await db.pending_leads.update_one(
            {"session_id": session_id},
            {"$set": {
                "session_id": session_id, "phone": phone,
                "whatsapp_name": contact.get("name") or "",
                "matched_properties": [item.get("title") for item in matches[:6]],
                "preferences": intent, "stage": "showing_results",
                "selected_property": None,
                "wacrm_contact_id": data.get("contact_id"),
                "wacrm_conversation_id": data.get("conversation_id"),
            }},
            upsert=True,
        )


async def run_due_followups() -> dict[str, int]:
    """Claim and send due campaign follow-ups without double-sending."""
    now = datetime.now(timezone.utc)
    sent = failed = stopped = 0
    for _ in range(100):
        campaign = await db.outbound_campaigns.find_one_and_update(
            {
                "status": "active",
                "next_followup_at": {"$lte": now},
                "$or": [
                    {"processing_until": {"$exists": False}},
                    {"processing_until": {"$lte": now}},
                ],
            },
            {"$set": {"processing_until": now + timedelta(minutes=10)}},
            sort=[("next_followup_at", 1)],
            return_document=ReturnDocument.AFTER,
        )
        if not campaign:
            break
        booked_lead = await db.leads.find_one({"$and": [
            {"$or": [
                {"whatsapp_phone": campaign["phone"]},
                {"phone": campaign["phone"]},
            ]},
            {"$or": [
                {"appointment_status": {"$in": ["requested", "scheduled", "confirmed", "booked"]}},
                {"lead_stage": {"$in": ["appointment_requested", "site_visit_scheduled", "booked"]}},
            ]},
        ]})
        if booked_lead:
            await db.outbound_campaigns.update_one(
                {"_id": campaign["_id"]},
                {"$set": {
                    "status": "converted", "converted_at": now,
                    "next_followup_at": None, "updated_at": now,
                }, "$unset": {"processing_until": ""}},
            )
            stopped += 1
            continue
        attempts = int(campaign.get("attempts", 0))
        if settings.followup_max_attempts > 0 and attempts >= settings.followup_max_attempts:
            await db.outbound_campaigns.update_one(
                {"_id": campaign["_id"]},
                {"$set": {"status": "completed", "next_followup_at": None}, "$unset": {"processing_until": ""}},
            )
            stopped += 1
            continue
        try:
            last_inbound = campaign.get("last_customer_message_at")
            within_service_window = bool(
                last_inbound and now - last_inbound <= timedelta(hours=24)
            )
            if within_service_window:
                body = await generate_sales_followup(f"wa:{campaign['phone']}")
                result = await send_text(campaign["phone"], body)
            else:
                template_name = settings.followup_template_name or campaign.get("initial_template_name")
                if not template_name:
                    raise RuntimeError("No approved follow-up template is configured")
                result = await send_template(
                    campaign["phone"], template_name,
                    campaign.get("template_language") or settings.followup_template_language or "en_US",
                    campaign.get("template_params"),
                )
            if not result:
                raise RuntimeError("WACRM did not return a sent message")
            sent += 1
            if campaign.get("submission_id"):
                await db.public_form_leads.update_one(
                    {"submission_id": campaign["submission_id"]},
                    {"$set": {
                        "status": "template_sent", "template_sent_at": now,
                        "wacrm_message_id": result.get("whatsapp_message_id"),
                        "conversation_id": result.get("conversation_id"),
                        "updated_at": now,
                    }},
                )
            await db.outbound_campaigns.update_one(
                {"_id": campaign["_id"]},
                {
                    "$set": {
                        "last_followup_at": now,
                        "last_followup_message_id": result.get("whatsapp_message_id"),
                        "next_followup_at": now + timedelta(hours=settings.followup_delay_hours),
                        "updated_at": now,
                    },
                    "$inc": {"attempts": 1},
                    "$unset": {"processing_until": "", "last_error": ""},
                },
            )
        except Exception as exc:
            failed += 1
            await db.outbound_campaigns.update_one(
                {"_id": campaign["_id"]},
                {
                    "$set": {
                        "last_error": str(exc)[:500],
                        "next_followup_at": now + timedelta(hours=1),
                    },
                    "$unset": {"processing_until": ""},
                },
            )
    return {"sent": sent, "failed": failed, "stopped": stopped}
