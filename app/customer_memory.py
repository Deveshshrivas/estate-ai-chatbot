"""Permanent customer identity and booking memory stored in MongoDB."""

from datetime import datetime, timezone
from typing import Any

from .database import db


async def save_customer_name(
    phone: str, name: str, source: str = "whatsapp", database: Any = None,
) -> None:
    store = database or db
    now = datetime.now(timezone.utc)
    await store.customer_profiles.update_one(
        {"phone": phone},
        {
            "$set": {
                "phone": phone,
                "name": name,
                "name_confirmed": True,
                "onboarding_stage": "complete",
                "name_source": source,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    await store.leads.update_many(
        {"$or": [{"whatsapp_phone": phone}, {"phone": phone}]},
        {"$set": {"name": name, "updated_at": now}},
    )


async def request_customer_name(phone: str, database: Any = None) -> None:
    store = database or db
    now = datetime.now(timezone.utc)
    await store.customer_profiles.update_one(
        {"phone": phone},
        {
            "$set": {
                "phone": phone,
                "onboarding_stage": "awaiting_name",
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


async def get_customer_profile(phone: str, database: Any = None) -> dict[str, Any]:
    store = database or db
    return await store.customer_profiles.find_one({"phone": phone}) or {}


async def remember_booking(
    *, phone: str, booking_type: str, status: str,
    property_title: str | None = None, scheduled_at: Any = None,
    contact_preference: str | None = None, source_id: str | None = None,
    database: Any = None,
) -> None:
    """Append a durable booking event instead of overwriting previous bookings."""
    now = datetime.now(timezone.utc)
    store = database or db
    await store.booking_history.insert_one({
        "phone": phone,
        "booking_type": booking_type,
        "status": status,
        "property_title": property_title,
        "scheduled_at": scheduled_at,
        "contact_preference": contact_preference,
        "source_id": source_id,
        "created_at": now,
        "updated_at": now,
    })


async def latest_booking(phone: str, database: Any = None) -> dict[str, Any]:
    store = database or db
    memory = await store.booking_history.find_one(
        {"phone": phone}, sort=[("created_at", -1)]
    )
    if memory:
        return memory
    # Backward-compatible memory for bookings made before booking_history existed.
    return await store.leads.find_one(
        {"$or": [{"whatsapp_phone": phone}, {"phone": phone}]},
        sort=[("updated_at", -1)],
    ) or {}


async def remember_cancellation(
    phone: str, reason: str, database: Any = None,
) -> None:
    store = database or db
    now = datetime.now(timezone.utc)
    latest = await store.booking_history.find_one(
        {"phone": phone, "status": {"$in": ["requested", "scheduled", "confirmed"]}},
        sort=[("created_at", -1)],
    )
    if latest:
        await store.booking_history.update_one(
            {"_id": latest["_id"]},
            {"$set": {
                "status": "cancelled", "cancellation_reason": reason,
                "cancelled_at": now, "updated_at": now,
            }},
        )
