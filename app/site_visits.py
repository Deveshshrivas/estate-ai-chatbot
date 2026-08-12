"""Shared salesperson availability and site-visit scheduling for the chatbot."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .database import db


INDIA = ZoneInfo("Asia/Kolkata")
ACTIVE_STATUSES = ["scheduled", "confirmed", "checked_in"]
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}


def parse_requested_slot(
    text: str, saved_date: str = "", now: datetime | None = None,
) -> tuple[datetime | None, bool, bool]:
    """Parse common customer date/time wording into an aware UTC datetime."""
    local_now = (now or datetime.now(INDIA)).astimezone(INDIA)
    content = " ".join(f"{saved_date} {text}".lower().split())
    date_value = None
    date_found = False
    if re.search(r"\btoday\b", content):
        date_value, date_found = local_now.date(), True
    elif re.search(r"\bday\s+after\s+to+m+or+o+w\b", content):
        date_value, date_found = (local_now + timedelta(days=2)).date(), True
    elif re.search(r"\bto+m+or+o+w\b", content):
        date_value, date_found = (local_now + timedelta(days=1)).date(), True
    else:
        iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", content)
        numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](20\d{2}|\d{2}))?\b", content)
        named = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+(20\d{2}))?\b", content)
        this_month = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?this\s+month\b", content)
        try:
            if iso:
                date_value = datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))).date()
                date_found = True
            elif numeric:
                year = int(numeric.group(3) or local_now.year)
                if year < 100:
                    year += 2000
                date_value = datetime(year, int(numeric.group(2)), int(numeric.group(1))).date()
                date_found = True
            elif named:
                month = MONTHS[named.group(2)]
                year = int(named.group(3) or local_now.year)
                date_value = datetime(year, month, int(named.group(1))).date()
                date_found = True
            elif this_month:
                date_value = datetime(local_now.year, local_now.month, int(this_month.group(1))).date()
                date_found = True
        except ValueError:
            return None, date_found, False
        if not date_found:
            for name, weekday in WEEKDAYS.items():
                if re.search(rf"\b{name[:3]}(?:{name[3:]})?\b", content):
                    days = (weekday - local_now.weekday()) % 7
                    date_value = (local_now + timedelta(days=days or 7)).date()
                    date_found = True
                    break
    if date_value and date_value < local_now.date() and not re.search(r"\b20\d{2}\b", content):
        try:
            date_value = date_value.replace(year=date_value.year + 1)
        except ValueError:
            return None, date_found, False

    time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", content)
    time_24 = re.search(r"\b(?:at\s+)?([01]?\d|2[0-3]):([0-5]\d)\b", content)
    at_hour = re.search(r"\bat\s+(\d{1,2})\b", content)
    time_found = bool(time_match or time_24 or at_hour)
    if not date_found or not time_found or not date_value:
        return None, date_found, time_found
    if time_match:
        hour, minute, meridiem = int(time_match.group(1)), int(time_match.group(2) or 0), time_match.group(3)
        if hour < 1 or hour > 12:
            return None, True, True
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    elif time_24:
        hour, minute = int(time_24.group(1)), int(time_24.group(2))
    else:
        hour, minute = int(at_hour.group(1)), 0
        if hour < 8:
            hour += 12
    local_slot = datetime(date_value.year, date_value.month, date_value.day, hour, minute, tzinfo=INDIA)
    return local_slot.astimezone(timezone.utc), True, True


async def available_salespeople(start: datetime, duration_minutes: int = 60) -> list[dict[str, Any]]:
    start = start.astimezone(timezone.utc)
    end = start + timedelta(minutes=duration_minutes)
    people = [person async for person in db.crm_users.find(
        {"role": "salesperson", "active": True},
        {"_id": 1, "full_name": 1, "email": 1},
    ).sort("full_name", 1)]
    visits = [visit async for visit in db.site_visits.find({
        "status": {"$in": ACTIVE_STATUSES}, "scheduled_at": {"$lt": end},
    })]
    busy_ids: set[str] = set()
    busy_names: set[str] = set()
    for visit in visits:
        visit_start = visit.get("scheduled_at")
        if not isinstance(visit_start, datetime):
            continue
        if visit_start.tzinfo is None:
            visit_start = visit_start.replace(tzinfo=timezone.utc)
        visit_end = visit_start + timedelta(minutes=int(visit.get("duration_minutes") or 60))
        if visit_end > start:
            busy_ids.add(str(visit.get("owner_id") or ""))
            busy_names.add(str(visit.get("owner") or "").casefold())
    return [
        {"id": str(person["_id"]), "name": person.get("full_name") or person.get("email"),
         "email": person.get("email", "")}
        for person in people
        if str(person["_id"]) not in busy_ids
        and str(person.get("full_name") or person.get("email") or "").casefold() not in busy_names
    ]


async def alternative_slots(start: datetime, limit: int = 3) -> list[datetime]:
    results: list[datetime] = []
    local_start = start.astimezone(INDIA)
    for day_offset in range(0, 8):
        day = (local_start + timedelta(days=day_offset)).date()
        if day.weekday() == 6:
            continue
        for hour in (10, 11, 12, 14, 15, 16, 17):
            candidate = datetime(day.year, day.month, day.day, hour, tzinfo=INDIA).astimezone(timezone.utc)
            if candidate <= start:
                continue
            if await available_salespeople(candidate):
                results.append(candidate)
                if len(results) >= limit:
                    return results
    return results


async def create_site_visit(
    *, phone: str, customer_name: str, customer_email: str, property_title: str,
    scheduled_at: datetime, source: str = "whatsapp",
) -> dict[str, Any]:
    available = await available_salespeople(scheduled_at)
    if not available:
        raise RuntimeError("No salesperson is available for this slot")
    owner = available[0]
    now = datetime.now(timezone.utc)
    record = {
        "customer_name": customer_name or phone, "customer_phone": phone,
        "customer_email": customer_email, "property_title": property_title,
        "scheduled_at": scheduled_at.astimezone(timezone.utc), "duration_minutes": 60,
        "status": "scheduled", "owner_id": owner["id"], "owner": owner["name"],
        "visit_type": "physical", "source": source, "created_at": now, "updated_at": now,
    }
    result = await db.site_visits.insert_one(record)
    return {**record, "id": str(result.inserted_id), "available_count": len(available)}


def display_slot(value: datetime) -> str:
    return value.astimezone(INDIA).strftime("%d %b %Y at %I:%M %p")
