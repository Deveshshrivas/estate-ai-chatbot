import hashlib
import hmac
import re
from urllib.parse import quote, urlparse
from datetime import datetime, timezone
from typing import Any
import httpx
from collections.abc import Awaitable, Callable
from pymongo.errors import DuplicateKeyError
from .agent import (
    _is_greeting, advise_on_selected_property, classify_booking_cancellation,
    classify_booking_type, classify_property_response, extract_customer_identity,
    extract_schedule, finalize_agent_answer, generate_advisor_reply,
    might_cancel_booking, graph,
)
from .config import get_settings
from .customer_memory import (
    get_customer_profile, latest_booking, remember_booking,
    remember_cancellation, request_customer_name, save_customer_name,
)
from .database import db, save_message
from .site_visits import (
    alternative_slots, available_salespeople, create_site_visit,
    display_slot, parse_requested_slot,
)

settings = get_settings()


def delivery_image_url(url: str) -> str:
    """Serve remote listing images through EstateAI for reliable Meta fetches."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return url
    if parsed.hostname == "images.unsplash.com":
        base = settings.app_url.rstrip("/")
        return f"{base}/api/property-image?url={quote(url, safe='')}"
    return url


def is_configured() -> bool:
    return bool(settings.whatsapp_access_token and settings.whatsapp_phone_number_id)


def verify_signature(body: bytes, signature: str | None) -> bool:
    if not settings.whatsapp_app_secret:
        return False
    expected = "sha256=" + hmac.new(
        settings.whatsapp_app_secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return bool(signature and hmac.compare_digest(expected, signature))


def extract_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            name = (value.get("contacts") or [{}])[0].get("profile", {}).get("name", "")
            for message in value.get("messages", []):
                message["_customer_name"] = name
                found.append(message)
    return found


async def _post(payload: dict) -> dict:
    url = (
        f"https://graph.facebook.com/{settings.whatsapp_graph_version}/"
        f"{settings.whatsapp_phone_number_id}/messages"
    )
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.post(
            url,
            headers={"Authorization": f"Bearer {settings.whatsapp_access_token}"},
            json={"messaging_product": "whatsapp", **payload},
        )
        response.raise_for_status()
        return response.json()


async def mark_read_and_typing(message_id: str) -> None:
    await _post({
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    })


async def send_text(to: str, body: str) -> None:
    # WhatsApp text bodies have a 4096-character limit.
    for start in range(0, len(body), 4000):
        await _post({"to": to, "type": "text", "text": {"body": body[start:start + 4000]}})


async def send_image(to: str, url: str, caption: str = "") -> None:
    image: dict[str, str] = {"link": url}
    if caption:
        image["caption"] = caption[:1024]
    await _post({"to": to, "type": "image", "image": image})


def _booking_memory_text(name: str, booking: dict[str, Any]) -> str:
    """Build a factual, customer-ready fallback from persistent booking data."""
    if not booking:
        return f"Hello {name} 👋 How can I help with your property search today?"
    status = str(booking.get("status") or booking.get("appointment_status") or "saved")
    property_title = booking.get("property_title")
    schedule = booking.get("scheduled_at") or booking.get("preferred_schedule")
    booking_type = str(booking.get("booking_type") or "consultation").replace("_", " ")
    details = booking_type
    if property_title:
        details += f" for {property_title}"
    if isinstance(schedule, datetime):
        details += f" on {display_slot(schedule)}"
    elif schedule:
        details += f" on {schedule}"
    return (
        f"Hello {name} 👋 I remember your {details}. Its current status is {status}. "
        "How can I help you today?"
    )


async def _handle_customer_identity(
    phone: str,
    text: str,
    send: Callable[[str, str], Awaitable[Any]] = send_text,
) -> bool:
    """Ask a name once, then greet using permanent customer and booking memory."""
    profile = (
        await get_customer_profile(phone, db)
        if hasattr(db, "customer_profiles") else {}
    )
    if profile.get("onboarding_stage") == "awaiting_name":
        identity = await extract_customer_identity(text)
        customer_name = str(identity.get("name") or "").strip()
        if not customer_name:
            reply = await generate_advisor_reply(
                "I didn't catch your name. What should I call you?",
                text,
                {"stage": "awaiting_name"},
            )
            await send(phone, reply)
            await save_message(f"wa:{phone}", "assistant", reply, channel="whatsapp")
            return True
        await save_customer_name(phone, customer_name, database=db)
        booking = await latest_booking(phone, db)
        required = _booking_memory_text(customer_name, booking)
        reply = await generate_advisor_reply(
            required, text,
            {"customer_name": customer_name, "latest_booking": booking},
        )
        await send(phone, reply)
        await save_message(f"wa:{phone}", "assistant", reply, channel="whatsapp")
        return True
    if not _is_greeting(text):
        return False
    customer_name = str(profile.get("name") or "").strip()
    if not profile.get("name_confirmed") or not customer_name:
        await request_customer_name(phone, db)
        reply = await generate_advisor_reply(
            "Hi! Welcome to Pratap AI Property Advisor. What name should I use for you?",
            text,
            {"stage": "new_customer"},
        )
        await send(phone, reply)
        await save_message(f"wa:{phone}", "assistant", reply, channel="whatsapp")
        return True
    booking = await latest_booking(phone, db)
    required = _booking_memory_text(customer_name, booking)
    reply = await generate_advisor_reply(
        required, text,
        {"customer_name": customer_name, "latest_booking": booking},
    )
    await send(phone, reply)
    await save_message(f"wa:{phone}", "assistant", reply, channel="whatsapp")
    return True


async def capture_engaged_lead(
    *,
    phone: str = "",
    name: str = "",
    intent: dict[str, Any] | None = None,
    matches: list[dict[str, Any]] | None = None,
    source: str = "whatsapp",
    session_id: str = "",
    last_message: str = "",
) -> bool:
    """Create a lead during discovery; appointment booking only enriches it later."""
    intent = intent or {}
    matches = matches or []
    if intent.get("intent") not in {"property_search", "seller_intake", "compare"} and not matches:
        return False
    identity = ({"whatsapp_phone": phone} if phone else {"session_id": session_id})
    if not phone and not session_id:
        return False
    existing = await db.leads.find_one(identity) or {}
    protected_statuses = {"qualified", "booked", "scheduled", "confirmed"}
    status = existing.get("status") if existing.get("status") in protected_statuses else "engaged"
    cities = intent.get("cities") if isinstance(intent.get("cities"), list) else []
    city = intent.get("city") or (cities[0] if cities else "")
    now = datetime.now(timezone.utc)
    details = {
        "name": name or existing.get("name") or ("Website visitor" if source == "web" else phone),
        "phone": existing.get("phone") or phone,
        "whatsapp_phone": phone or existing.get("whatsapp_phone"),
        "session_id": session_id or existing.get("session_id") or (f"wa:{phone}" if phone else ""),
        "source": source,
        "status": status,
        "lead_stage": "discovery",
        "appointment_status": existing.get("appointment_status") or "not_booked",
        "property_purpose": intent.get("purpose") or existing.get("property_purpose"),
        "city": city or existing.get("city"),
        "cities": cities or existing.get("cities", []),
        "property_type": intent.get("property_type") or intent.get("type") or existing.get("property_type"),
        "bedrooms": intent.get("bedrooms") or intent.get("min_bedrooms") or existing.get("bedrooms"),
        "budget_min": intent.get("min_price") or intent.get("budget_min") or existing.get("budget_min"),
        "budget_max": intent.get("max_price") or intent.get("budget_max") or existing.get("budget_max"),
        "budget_currency": "INR",
        "matched_properties": [item.get("title") for item in matches if item.get("title")][:6],
        "last_customer_message": last_message,
        "updated_at": now,
    }
    await db.leads.update_one(
        identity,
        {"$set": details, "$setOnInsert": {"created_at": now}, "$inc": {"engagement_count": 1}},
        upsert=True,
    )
    return True


async def stop_sales_followups(phone: str, status: str = "converted") -> None:
    """Stop every automated campaign after a booking or explicit opt-out."""
    now = datetime.now(timezone.utc)
    await db.outbound_campaigns.update_many(
        {"phone": phone, "status": "active"},
        {"$set": {
            "status": status,
            f"{status}_at": now,
            "next_followup_at": None,
            "updated_at": now,
        }},
    )
    await db.public_form_leads.update_many(
        {"phone": phone, "status": {"$in": ["template_scheduled", "template_sent", "engaged"]}},
        {"$set": {"status": status, "updated_at": now}},
    )


async def process_message(message: dict[str, Any]) -> None:
    message_id = message.get("id", "")
    try:
        await db.webhook_events.insert_one({
            "message_id": message_id, "created_at": datetime.now(timezone.utc)
        })
    except DuplicateKeyError:
        return

    sender = message.get("from", "")
    message_type = message.get("type")
    if message_type != "text":
        await send_text(sender, "Please send your property request as text. Photo analysis is available in the EstateAI web chat.")
        return
    text = message.get("text", {}).get("body", "").strip()
    if not text:
        return

    try:
        await mark_read_and_typing(message_id)
        await save_message(f"wa:{sender}", "user", text, channel="whatsapp")
        if await _handle_customer_identity(sender, text):
            return
        if await _cancel_saved_booking(sender, text, ai_replies=True):
            return
        if await _handle_lead_details(sender, message.get("_customer_name", ""), text, ai_replies=True):
            return
        result = await graph.ainvoke({
            "session_id": f"wa:{sender}",
            "message": text,
            "image": None,
        })
        answer = await finalize_agent_answer(
            result,
            text,
            {
                "intent": result.get("intent", {}),
                "properties": [
                    item.get("title") for item in result.get("matches", [])
                ],
            },
        )
        intent = result.get("intent", {})
        matches = result.get("matches", [])
        focused_property = bool((result.get("page_info") or {}).get("focused_property"))
        await capture_engaged_lead(
            phone=sender,
            name=message.get("_customer_name", ""),
            intent=intent,
            matches=matches,
            last_message=text,
        )
        # Property-search results should feel visual by default. An explicit photo
        # request receives a larger gallery; an ordinary matched search receives
        # the primary photo for its best three results.
        should_send_images = bool(matches) and (
            intent.get("intent") in {"property_search", "compare"}
            or intent.get("wants_images")
        )
        if should_send_images:
            sent = 0
            # The search layer already caps pages at five for one city or
            # three per city for a two-city comparison.
            listing_limit = len(matches)
            # Keep each result atomic: one primary image paired with that same
            # property's facts, then move on to the next matching property.
            images_per_listing = 20 if focused_property and intent.get("wants_images") else 1
            caption_cards = []
            for listing in matches[:listing_limit]:
                fallback = (
                    f"🏡 {listing['title']}\n"
                    f"📍 {listing['locality']}, {listing['city']}\n"
                    f"{listing.get('bedrooms', 0)} BHK · "
                    f"{listing.get('area_sqft', 0):,} sq ft\n"
                    f"💰 {_format_price(listing)}\n"
                    f"🛋️ {listing.get('furnishing', 'Furnishing on request')}\n"
                    f"✨ {', '.join(listing.get('amenities', [])[:3])}\n\n"
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
            for listing, caption in zip(matches[:listing_limit], captions):
                for image_url in listing.get("images", [])[:images_per_listing]:
                    await send_image(sender, delivery_image_url(image_url), caption)
                    sent += 1
            if intent.get("wants_images") and not sent:
                await send_text(sender, "I don’t have matching property photos yet. Tell me your city, budget and BHK so I can narrow it down.")
        await send_text(sender, answer)
        await save_message(f"wa:{sender}", "assistant", answer, channel="whatsapp")

        if matches:
            await db.pending_leads.update_one(
                {"session_id": f"wa:{sender}"},
                {"$set": {
                    "session_id": f"wa:{sender}", "phone": sender,
                    "whatsapp_name": message.get("_customer_name", ""),
                    "matched_properties": [x.get("title") for x in matches[:6]],
                    "preferences": intent, "stage": "showing_results",
                    "selected_property": None,
                }},
                upsert=True,
            )
    except Exception as exc:
        await db.webhook_errors.insert_one({
            "message_id": message_id, "error": str(exc),
            "created_at": datetime.now(timezone.utc),
        })
        try:
            await send_text(sender, "Sorry, I hit a temporary issue. Please try again in a moment.")
        except Exception:
            pass


async def _handle_lead_details(
    phone: str,
    name: str,
    text: str,
    send: Callable[[str, str], Awaitable[Any]] = send_text,
    send_property_image: Callable[[str, str, str], Awaitable[Any]] = send_image,
    ai_replies: bool = False,
) -> bool:
    """Keep Q&A open and collect details only after booking consent."""
    lowered = text.lower()
    pending = await db.pending_leads.find_one({"session_id": f"wa:{phone}"})
    if not pending:
        return False
    profile = (
        await get_customer_profile(phone, db)
        if hasattr(db, "customer_profiles") else {}
    )
    confirmed_name = (
        str(profile.get("name") or "").strip()
        if profile.get("name_confirmed") else ""
    )
    if confirmed_name:
        name = confirmed_name
    if ai_replies:
        raw_send = send

        async def send(recipient: str, required_message: str) -> Any:
            reply = await generate_advisor_reply(
                required_message,
                text,
                {
                    "stage": pending.get("stage"),
                    "selected_property": pending.get("selected_property"),
                    "customer_name": name or pending.get("whatsapp_name"),
                },
            )
            return await raw_send(recipient, reply)
    properties = pending.get("matched_properties", [])
    stage = pending.get("stage", "showing_results")
    if ai_replies and _is_greeting(text) and stage.startswith("awaiting_"):
        customer_name = name or pending.get("whatsapp_name") or "there"
        property_title = pending.get("selected_property")
        subject = (
            f" for {property_title}" if property_title
            else " with our property advisor"
        )
        await send(
            phone,
            f"Hi {customer_name}. Your earlier request{subject} is still in progress. "
            "Would you like to continue?",
        )
        return True
    if stage == "awaiting_booking_type":
        booking_type = await classify_booking_type(text)
        if not booking_type:
            await send(
                phone,
                "Would you prefer a site visit or an advisor appointment by phone/video call?",
            )
            return True
        next_stage = (
            "awaiting_site_visit_schedule"
            if booking_type == "site_visit"
            else "awaiting_booking_email"
        )
        chosen_preference = None
        if booking_type == "appointment":
            chosen_preference = (
                "video_call" if re.search(r"\b(video|meet|zoom)\b", lowered)
                else "phone_call" if re.search(
                    r"\b(phone|voice|callback|call me|whatsapp call)\b", lowered
                ) else None
            )
        booking_fields: dict[str, Any] = {
            "booking_type": booking_type,
        }
        if chosen_preference:
            booking_fields["contact_preference"] = chosen_preference
        if not confirmed_name:
            await db.pending_leads.update_one(
                {"_id": pending["_id"]},
                {"$set": {
                    "stage": "awaiting_booking_name",
                    "after_name_stage": next_stage,
                    **booking_fields,
                }},
            )
            await send(phone, "Great. What name should I use for the booking?")
            return True
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {
                "stage": next_stage,
                **booking_fields,
                "customer_name": confirmed_name,
                "contact_phone": phone,
            }},
        )
        if booking_type == "site_visit":
            await send(phone, "What date and time would you like to visit the property?")
        else:
            await send(phone, "What email should I use for the appointment confirmation?")
        return True
    if stage == "awaiting_booking_name":
        identity = await extract_customer_identity(text)
        customer_name = str(identity.get("name") or "").strip()
        if not customer_name:
            await send(phone, "What name should I use for your booking?")
            return True
        await save_customer_name(phone, customer_name, database=db)
        name = customer_name
        next_stage = str(pending.get("after_name_stage") or "awaiting_booking_email")
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {
                "stage": next_stage,
                "customer_name": customer_name,
                "contact_phone": phone,
            }},
        )
        if next_stage == "awaiting_site_visit_schedule":
            await send(
                phone,
                f"Hello {customer_name} 👋 What date and time would you like to visit the property?",
            )
        else:
            await send(phone, f"Hello {customer_name} 👋 What email should I use for the confirmation?")
        return True
    inline_parts = [part.strip() for part in text.split("|")]
    has_inline_name = (
        len(inline_parts) > 1 and bool(inline_parts[0])
        and "@" not in inline_parts[0]
        and not any(character.isdigit() for character in inline_parts[0])
    )
    if stage == "awaiting_details" and not confirmed_name and not has_inline_name:
        await db.pending_leads.update_one(
            {"_id": pending["_id"]}, {"$set": {"stage": "awaiting_booking_name"}}
        )
        await send(phone, "Before I arrange it, what name should I use for your booking?")
        return True
    if stage in {"awaiting_details", "awaiting_booking_email"}:
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        if not email:
            await send(phone, "What email should I use to send the confirmation?")
            return True
        parts = [part.strip() for part in text.split("|")]
        supplied_name = parts[0] if len(parts) > 1 and "@" not in parts[0] else ""
        supplied_phone = re.search(r"(?:\+?\d[\d\s-]{7,}\d)", text)
        inline_preference = "video_call" if re.search(r"\b(video|meet|zoom)\b", lowered) else (
            "phone_call" if re.search(r"\b(phone|voice|call|whatsapp call)\b", lowered) else None
        )
        customer_name = supplied_name or name or pending.get("whatsapp_name") or "there"
        preference = pending.get("contact_preference") or inline_preference
        next_stage = "awaiting_call_schedule" if preference else "awaiting_booking_preference"
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {
                "stage": next_stage,
                "customer_name": customer_name,
                "email": email.group(0),
                "contact_phone": supplied_phone.group(0) if supplied_phone else phone,
            }},
        )
        if preference:
            method = "video meeting" if preference == "video_call" else "phone call"
            await send(phone, f"Thanks, {customer_name}. What date and time works for the {method}?")
        else:
            await send(phone, "Thanks. Would you prefer a phone call or a video meeting?")
        return True
    if stage == "awaiting_booking_preference":
        preference = "video_call" if re.search(r"\b(video|meet|zoom)\b", lowered) else (
            "phone_call" if re.search(r"\b(phone|voice|call|whatsapp call)\b", lowered) else None
        )
        if not preference:
            await send(phone, "Would a phone call or a video meeting work better for you?")
            return True
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {
                "stage": "awaiting_call_schedule",
                "contact_preference": preference,
                "contact_phone": phone,
            }},
        )
        method = "video meeting" if preference == "video_call" else "phone call"
        await send(phone, f"Perfect. What date and time works for the {method}?")
        return True
    if stage == "awaiting_site_visit_schedule":
        saved_date = str(pending.get("preferred_date") or "")
        slot, has_date, has_time = (
            await extract_schedule(text, saved_date)
            if ai_replies else parse_requested_slot(text, saved_date)
        )
        if not has_date:
            await send(phone, "What date would you like to visit the property?")
            return True
        if not has_time:
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": {"preferred_date": text}}
            )
            await send(phone, "What time works on that date?")
            return True
        if not slot or slot <= datetime.now(timezone.utc):
            await send(phone, "Please choose a future date and time for the site visit.")
            return True
        available = await available_salespeople(slot)
        if not available:
            alternatives = await alternative_slots(slot)
            choices = ", ".join(display_slot(item) for item in alternatives)
            await send(
                phone,
                "No salesperson is free at that time. "
                + (f"Available alternatives: {choices}. Which one works?" if choices else "Please choose another date and time."),
            )
            return True
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {
                "stage": "awaiting_site_visit_email",
                "preferred_schedule": slot,
                "available_salespeople": len(available),
            }},
        )
        await send(
            phone,
            f"{len(available)} salesperson{'s are' if len(available) != 1 else ' is'} available on {display_slot(slot)}. What email should we use for confirmation?",
        )
        return True
    if stage == "awaiting_site_visit_email":
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        if not email:
            await send(phone, "Please share a valid email for the site-visit confirmation.")
            return True
        slot = pending.get("preferred_schedule")
        if not isinstance(slot, datetime):
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": {"stage": "awaiting_site_visit_schedule"}}
            )
            await send(phone, "Please send the site-visit date and time again.")
            return True
        if slot.tzinfo is None:
            slot = slot.replace(tzinfo=timezone.utc)
        available = await available_salespeople(slot)
        if not available:
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": {"stage": "awaiting_site_visit_schedule"}}
            )
            alternatives = await alternative_slots(slot)
            choices = ", ".join(display_slot(item) for item in alternatives)
            await send(phone, f"That slot was just booked. Available alternatives: {choices}. Which one works?")
            return True
        visit = await create_site_visit(
            phone=phone,
            customer_name=name or pending.get("whatsapp_name") or phone,
            customer_email=email.group(0),
            property_title=pending.get("selected_property") or "Selected property",
            scheduled_at=slot,
        )
        now = datetime.now(timezone.utc)
        await db.leads.update_one(
            {"whatsapp_phone": phone},
            {"$set": {
                "name": name or pending.get("whatsapp_name") or phone,
                "email": email.group(0), "phone": phone, "source": "whatsapp",
                "status": "scheduled", "lead_stage": "site_visit_scheduled",
                "booking_type": "site_visit",
                "appointment_status": "scheduled", "preferred_schedule": slot,
                "site_visit_id": visit["id"], "assigned_to": visit["owner_id"],
                "assigned_name": visit["owner"], "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        if hasattr(db, "booking_history"):
            await remember_booking(
                phone=phone, booking_type="site_visit", status="scheduled",
                property_title=pending.get("selected_property") or "Selected property",
                scheduled_at=slot, source_id=str(visit.get("id") or ""),
                customer_name=name or pending.get("whatsapp_name") or phone,
                database=db,
            )
        await stop_sales_followups(phone)
        await db.pending_leads.delete_one({"_id": pending["_id"]})
        await send(
            phone,
            f"Your site visit is confirmed for {display_slot(slot)} with {visit['owner']}. We have saved it in the CRM calendar.",
        )
        return True
    if stage == "awaiting_call_schedule":
        saved_date = str(pending.get("preferred_date") or "")
        slot, has_date, has_time = (
            await extract_schedule(text, saved_date)
            if ai_replies else parse_requested_slot(text, saved_date)
        )
        if not has_date and not saved_date:
            await send(phone, "What date works for the callback?")
            return True
        if not has_time:
            if has_date:
                await db.pending_leads.update_one(
                    {"_id": pending["_id"]}, {"$set": {"preferred_date": text}}
                )
            await send(phone, "What time works on that date?")
            return True
        if not slot or slot <= datetime.now(timezone.utc):
            await send(phone, "Please choose a future date and time for the callback.")
            return True
        schedule = slot
        schedule_label = display_slot(slot)
        if pending.get("email") and pending.get("customer_name"):
            now = datetime.now(timezone.utc)
            preference = pending.get("contact_preference") or "phone_call"
            await db.leads.update_one(
                {"whatsapp_phone": phone},
                {"$set": {
                    "name": pending["customer_name"], "email": pending["email"],
                    "phone": pending.get("contact_phone") or phone,
                    "whatsapp_phone": phone, "contact_preference": preference,
                    "source": "whatsapp", "status": "qualified",
                    "lead_stage": "appointment_requested",
                    "booking_type": "appointment",
                    "appointment_status": "requested",
                    "preferred_schedule": schedule,
                    "property_title": pending.get("selected_property"),
                    "matched_properties": properties,
                    "preferences": pending.get("preferences", {}), "updated_at": now,
                }, "$setOnInsert": {"created_at": now}, "$inc": {"enquiry_count": 1}},
                upsert=True,
            )
            if hasattr(db, "booking_history"):
                await remember_booking(
                    phone=phone, booking_type="consultation", status="requested",
                    property_title=pending.get("selected_property"),
                    scheduled_at=schedule, contact_preference=preference,
                    customer_name=pending.get("customer_name") or name or phone,
                    database=db,
                )
            await db.pending_leads.delete_one({"_id": pending["_id"]})
            await stop_sales_followups(phone)
            method = "video call" if preference == "video_call" else "phone call"
            await send(
                phone,
                f"Thank you, {pending['customer_name']}! Your {method} request for "
                f"{schedule_label} is saved. Our property advisor will confirm it shortly.",
            )
            return True
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {"stage": "awaiting_call_email", "preferred_schedule": schedule}},
        )
        await send(phone, "Got it. What email should we use for the appointment confirmation?")
        return True
    if stage == "awaiting_call_email":
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        if not email:
            await send(phone, "Please share a valid email for the appointment confirmation.")
            return True
        now = datetime.now(timezone.utc)
        await db.leads.update_one(
            {"whatsapp_phone": phone},
            {"$set": {
                "name": name or pending.get("whatsapp_name") or phone,
                "email": email.group(0), "phone": phone, "whatsapp_phone": phone,
                "contact_preference": pending.get("contact_preference") or "phone_call",
                "source": "whatsapp",
                "status": "qualified", "lead_stage": "appointment_requested",
                "booking_type": "appointment",
                "appointment_status": "requested",
                "preferred_schedule": pending.get("preferred_schedule"),
                "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        if hasattr(db, "booking_history"):
            await remember_booking(
                phone=phone, booking_type="consultation", status="requested",
                property_title=pending.get("selected_property"),
                scheduled_at=pending.get("preferred_schedule"),
                contact_preference=pending.get("contact_preference") or "phone_call",
                customer_name=name or pending.get("whatsapp_name") or phone,
                database=db,
            )
        await stop_sales_followups(phone)
        await db.pending_leads.delete_one({"_id": pending["_id"]})
        saved_schedule = pending.get("preferred_schedule")
        schedule_label = (
            display_slot(saved_schedule) if isinstance(saved_schedule, datetime)
            else str(saved_schedule or "the requested time")
        )
        await send(
            phone,
            f"Your callback request for {schedule_label} is saved. "
            "Our property advisor will confirm it shortly.",
        )
        return True
    if stage in {"showing_results", "awaiting_property"}:
        try:
            decision = await classify_property_response(text, properties)
        except Exception:
            # A model outage must not lose an explicit listing-name selection.
            selected = next(
                (title for title in properties if title.casefold() in text.casefold()),
                None,
            )
            decision = {"is_interested": bool(selected), "selected_property": selected}
        if not decision["is_interested"] and stage == "showing_results":
            return False
        selected = decision.get("selected_property")
        if not selected:
            choices = "\n".join(f"{i + 1}. {title}" for i, title in enumerate(properties))
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": {"stage": "awaiting_property"}}
            )
            await send(phone, f"Which property interests you? Reply with its name or number:\n{choices}")
            return True
        await db.pending_leads.update_one(
            {"_id": pending["_id"]},
            {"$set": {"stage": "interested", "selected_property": selected}},
        )
        now = datetime.now(timezone.utc)
        await db.leads.update_one(
            {"whatsapp_phone": phone},
            {
                "$set": {
                    "name": name or pending.get("whatsapp_name") or phone,
                    "phone": phone,
                    "whatsapp_phone": phone,
                    "source": "whatsapp",
                    "status": "interested",
                    "lead_stage": "property_selected",
                    "appointment_status": "not_booked",
                    "property_title": selected,
                    "matched_properties": properties,
                    "preferences": pending.get("preferences", {}),
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        if decision.get("is_question"):
            listing = await db.properties.find_one({"title": selected}, {"_id": 0}) or {
                "title": selected
            }
            try:
                advice = await advise_on_selected_property(text, listing)
            except Exception:
                advice = {
                    "answer": "I can help with this property, but that detail needs confirmation.",
                    "wants_images": False,
                }
            if advice.get("wants_images"):
                for index, image_url in enumerate(listing.get("images", []), start=1):
                    await send_property_image(
                        phone, delivery_image_url(image_url),
                        f"{listing.get('title', selected)} — image {index}",
                    )
            answer = advice.get("answer") or "What would you like to know about this property?"
            await send(phone, f"{answer}\n\nWould you like to book a consultation for it?")
            return True
        await send(
            phone,
            f"Great choice—{selected}. Would you like to book a consultation, "
            "or ask me anything about this property?"
        )
        return True

    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(?:\+?\d[\d\s-]{7,}\d)", text)
    preference = "video_call" if "video" in lowered else (
        "phone_call" if any(word in lowered for word in ("phone", "call", "voice")) else None
    )
    if stage in {"interested", "awaiting_details"} and not (
        email_match and phone_match and preference
    ):
        selected = pending.get("selected_property")
        listing = await db.properties.find_one({"title": selected}, {"_id": 0}) or {
            "title": selected
        }
        try:
            decision = await advise_on_selected_property(text, listing)
        except Exception:
            decision = {
                "action": "other", "wants_images": False,
                "answer": "I can help with details about this property.",
            }
        if decision["action"] == "book":
            booking_type = await classify_booking_type(text)
            if not booking_type:
                await db.pending_leads.update_one(
                    {"_id": pending["_id"]},
                    {"$set": {"stage": "awaiting_booking_type"}},
                )
                await send(
                    phone,
                    "Would you prefer a site visit or an advisor appointment by phone/video call?",
                )
                return True
            wants_site_visit = booking_type == "site_visit"
            if not confirmed_name:
                await db.pending_leads.update_one(
                    {"_id": pending["_id"]},
                    {"$set": {
                        "stage": "awaiting_booking_name",
                        "after_name_stage": (
                            "awaiting_site_visit_schedule" if wants_site_visit
                            else "awaiting_booking_email"
                        ),
                        "booking_type": booking_type,
                    }},
                )
                await send(phone, "Great—before I arrange it, what name should I use for your booking?")
                return True
            if wants_site_visit:
                await db.pending_leads.update_one(
                    {"_id": pending["_id"]},
                    {"$set": {"stage": "awaiting_site_visit_schedule", "booking_type": "site_visit"}},
                )
                slot, has_date, has_time = (
                    await extract_schedule(text) if ai_replies else parse_requested_slot(text)
                )
                if slot and has_date and has_time:
                    available = await available_salespeople(slot)
                    if available:
                        await db.pending_leads.update_one(
                            {"_id": pending["_id"]},
                            {"$set": {"stage": "awaiting_site_visit_email", "preferred_schedule": slot, "available_salespeople": len(available)}},
                        )
                        await send(phone, f"{len(available)} salesperson{'s are' if len(available) != 1 else ' is'} available on {display_slot(slot)}. What email should we use for confirmation?")
                        return True
                    alternatives = await alternative_slots(slot)
                    choices = ", ".join(display_slot(item) for item in alternatives)
                    await send(phone, f"No salesperson is free then. Available alternatives: {choices}. Which one works?")
                    return True
                await send(phone, "Sure. What date and time would you like to visit the property?")
                return True
            booking_changes: dict[str, Any] = {
                "stage": "awaiting_booking_email" if confirmed_name else "awaiting_booking_name",
                "customer_name": confirmed_name or None,
                "contact_phone": phone,
            }
            if preference:
                booking_changes["contact_preference"] = preference
            await db.pending_leads.update_one(
                {"_id": pending["_id"]}, {"$set": booking_changes}
            )
            if confirmed_name:
                await send(
                    phone,
                    f"Great, {confirmed_name}—I can arrange that. What email should I use for the confirmation?",
                )
            else:
                await send(phone, "Great—I can arrange that. What name should I use for your booking?")
            return True
        if decision["action"] == "cancel":
            await db.pending_leads.update_one(
                {"_id": pending["_id"]},
                {"$set": {
                    "stage": "interested", "booking_cancelled_at": datetime.now(timezone.utc),
                }},
            )
            await send(
                phone,
                f"Your consultation request for {selected} is cancelled. "
                "You can continue asking questions or book again anytime.",
            )
            return True
        if decision["action"] == "decline":
            await send(phone, "No problem. Ask me about this property or explore another one.")
            return True
        if decision.get("wants_images"):
            for index, image_url in enumerate(listing.get("images", []), start=1):
                await send_property_image(
                    phone, delivery_image_url(image_url),
                    f"{listing.get('title', selected)} — image {index}",
                )
        answer = decision.get("answer") or "What would you like to know about this property?"
        await send(
            phone, f"{answer}\n\nWould you like to book a consultation for this property?"
        )
        return True

    return False


async def _cancel_saved_booking(
    phone: str,
    text: str,
    send: Callable[[str, str], Awaitable[Any]] = send_text,
    ai_replies: bool = False,
) -> bool:
    """Cancel the newest active consultation request without deleting its audit record."""
    if not might_cancel_booking(text):
        return False
    booking = await db.leads.find_one(
        {
            "whatsapp_phone": phone,
            "$or": [
                {"appointment_status": {"$in": ["requested", "scheduled", "confirmed"]}},
                {"status": {"$in": ["booked", "scheduled", "confirmed"]}},
            ],
        },
        sort=[("updated_at", -1)],
    )
    if not booking:
        return False
    try:
        should_cancel = await classify_booking_cancellation(text, booking)
    except Exception:
        return False
    if not should_cancel:
        return False
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
    await db.outbound_campaigns.update_many(
        {"phone": phone, "status": {"$in": ["active", "converted"]}},
        {"$set": {"status": "cancelled", "cancelled_at": now, "next_followup_at": None}},
    )
    await db.site_visits.update_many(
        {"customer_phone": phone, "status": {"$in": ["scheduled", "confirmed"]}},
        {"$set": {
            "status": "cancelled", "cancelled_at": now,
            "cancellation_reason": text, "updated_at": now,
        }},
    )
    if hasattr(db, "booking_history"):
        await remember_cancellation(phone, text, db)
    property_title = booking.get("property_title") or "your selected property"
    confirmation = (
        f"Your consultation request for {property_title} has been cancelled. "
        "You can book again anytime."
    )
    if ai_replies:
        confirmation = await generate_advisor_reply(
            confirmation,
            text,
            {"property_title": property_title, "appointment_status": "cancelled"},
        )
    await send(
        phone,
        confirmation,
    )
    return True


def _format_price(listing: dict[str, Any]) -> str:
    price = float(listing.get("price", 0))
    if listing.get("purpose") == "rent":
        return f"₹{price:,.0f}/month"
    if price >= 10_000_000:
        return f"₹{price / 10_000_000:.2f} crore"
    if price >= 100_000:
        return f"₹{price / 100_000:.1f} lakh"
    return f"₹{price:,.0f}"
