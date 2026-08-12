"""Seed and verify one persistent, clearly labelled end-to-end CRM demo journey.

This script never sends WhatsApp messages. It stores the records in the configured
EstateAI MongoDB database so Pratap One can display and manage them afterwards.
It is idempotent: rerunning it updates the same E2E records instead of duplicating them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agent import graph
from app.database import client, db, initialize_database, save_message
from app.whatsapp import _cancel_saved_booking


RUN_KEY = "pratap-e2e-demo-20260810"
SESSION_ID = f"e2e:{RUN_KEY}"
PHONE = "919900001010"
CUSTOMER = "Aarav E2E Customer"
EMAIL = "aarav.e2e@example.com"


def disposable_password_hash() -> str:
    """Create a valid CRM scrypt hash without exposing a reusable test password."""
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(
        secrets.token_bytes(32), salt=salt.encode(), n=16384, r=8, p=1, dklen=64
    )
    return f"{salt}:{digest.hex()}"


async def upsert(collection: str, key: str, record: dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    result = await db[collection].find_one_and_update(
        {"e2e_key": key},
        {
            "$set": {**record, "e2e_key": key, "updated_at": now},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
        return_document=True,
    )
    return str(result["_id"])


async def seed_properties() -> list[dict[str, Any]]:
    common = {
        "currency": "INR",
        "purpose": "buy",
        "active": True,
        "verified": True,
        "inventory_status": "available",
        "agent": {"name": "E2E Sales Advisor", "phone": "+91 99000 01010", "email": "advisor.e2e@example.com"},
    }
    rows = [
        {
            **common,
            "title": "Pratap E2E Lakeview Flat",
            "project_name": "Pratap E2E Residency",
            "developer": "Pratap Demo Developers",
            "tower_name": "Tower A",
            "unit_number": "A-1204",
            "city": "Bengaluru",
            "locality": "Whitefield",
            "type": "Flat",
            "bedrooms": 2,
            "bathrooms": 2,
            "area_sqft": 1180,
            "price": 8200000,
            "amenities": ["Metro nearby", "Clubhouse", "Parking", "Power backup"],
            "nearby": ["Kadugodi Tree Park Metro", "Hospital", "School"],
            "coordinates": {"latitude": 12.9698, "longitude": 77.7500},
            "description": "E2E demo 2 BHK flat with stored price, location and amenity evidence.",
            "images": ["https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?auto=format&fit=crop&w=1200&q=80"],
        },
        {
            **common,
            "title": "Pratap E2E Independent House",
            "project_name": "",
            "tower_name": "",
            "unit_number": "House 18",
            "city": "Bengaluru",
            "locality": "HSR Layout",
            "type": "Independent house",
            "bedrooms": 3,
            "bathrooms": 3,
            "area_sqft": 2100,
            "price": 18500000,
            "amenities": ["Private parking", "Terrace", "Solar water heater"],
            "coordinates": {"latitude": 12.9116, "longitude": 77.6474},
            "description": "E2E demo standalone house, intentionally not attached to a tower.",
            "images": ["https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1200&q=80"],
        },
        {
            **common,
            "title": "Pratap E2E Residential Plot",
            "project_name": "Pratap E2E Green Acres",
            "tower_name": "",
            "unit_number": "Plot 21",
            "city": "Bengaluru",
            "locality": "Sarjapur Road",
            "type": "Residential plot",
            "bedrooms": 0,
            "bathrooms": 0,
            "area_sqft": 2400,
            "price": 9600000,
            "amenities": ["Road access", "Gated layout", "Electricity connection"],
            "coordinates": {"latitude": 12.9008, "longitude": 77.6913},
            "description": "E2E demo plot with project and plot-number inventory hierarchy.",
            "images": ["https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80"],
        },
    ]
    for index, row in enumerate(rows, 1):
        row["id"] = await upsert("properties", f"{RUN_KEY}:property:{index}", row)
    return rows


async def seed_crm_journey(properties: list[dict[str, Any]]) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    visit_at = now + timedelta(days=4)
    property_row = properties[0]

    salesperson_id = await upsert(
        "crm_users",
        f"{RUN_KEY}:salesperson",
        {
            "full_name": "E2E Sales Advisor",
            "email": "advisor.e2e@example.com",
            "password_hash": disposable_password_hash(),
            "role": "salesperson",
            "active": True,
        },
    )
    lead_profile = {
        "company": "Aarav Homes Pvt Ltd",
        "property_purpose": "buy",
        "city": "Bengaluru",
        "localities": ["Whitefield", "Kadugodi"],
        "property_type": "Flat",
        "bedrooms": 2,
        "budget_min": 7000000,
        "budget_max": 10000000,
        "budget_currency": "INR",
        "timeline": "Within 3 months",
        "financing": "Home loan pre-approved",
        "query_reason": "Need a 2 BHK near metro for self-use with parking and power backup.",
        "platform_name": "Pratap E2E public form + WhatsApp",
        "form_name": "E2E Property Buyer Qualification",
        "score": 85,
        "priority": "High",
        "assigned_to": salesperson_id,
        "assigned_name": "E2E Sales Advisor",
        "follow_up_status": "cancelled",
        "follow_up_reason": "Customer cancelled the test consultation",
        "last_follow_up_at": now,
        "follow_up_completed_by": "E2E Sales Advisor",
        "notes": "Complete persistent dummy lead used to validate every CRM field.",
        "tags": ["E2E", "2 BHK", "near metro", "loan ready"],
    }
    lead_id = await upsert(
        "leads",
        f"{RUN_KEY}:lead",
        {
            "name": CUSTOMER,
            "phone": PHONE,
            "whatsapp_phone": PHONE,
            "email": EMAIL,
            **lead_profile,
            "source": "e2e_saved_demo",
            "property_title": property_row["title"],
            "matched_properties": [item["title"] for item in properties],
            "status": "qualified",
            "lead_stage": "appointment_requested",
            "appointment_status": "requested",
            "preferred_schedule": visit_at,
            "contact_preference": "phone_call",
        },
    )
    visit_id = await upsert(
        "site_visits",
        f"{RUN_KEY}:visit",
        {
            "customer_name": CUSTOMER,
            "customer_phone": PHONE,
            "customer_email": EMAIL,
            "property_id": property_row["id"],
            "property_title": property_row["title"],
            "scheduled_at": visit_at,
            "duration_minutes": 60,
            "status": "scheduled",
            "owner_id": salesperson_id,
            "owner": "E2E Sales Advisor",
            "visit_type": "physical",
            "source": "e2e_saved_demo",
        },
    )
    booking_id = await upsert(
        "bookings",
        f"{RUN_KEY}:booking",
        {
            "booking_number": "BK-E2E-20260810",
            "customer_name": CUSTOMER,
            "customer_phone": PHONE,
            "customer_email": EMAIL,
            "property_id": property_row["id"],
            "property_title": property_row["title"],
            "unit": "A-1204",
            "booking_value": 8200000,
            "amount_paid": 100000,
            "currency": "INR",
            "status": "draft",
            "kyc_status": "pending",
            "documents_status": "pending",
            "agreement_status": "not_started",
            "owner": "E2E Sales Advisor",
            "source": "e2e_saved_demo",
        },
    )
    await upsert(
        "outbound_campaigns",
        f"{RUN_KEY}:campaign",
        {
            "phone": PHONE,
            "contact_name": CUSTOMER,
            "status": "active",
            "template_name": "e2e_property_followup",
            "next_followup_at": now + timedelta(hours=5),
            "source": "e2e_saved_demo",
        },
    )
    await upsert(
        "crm_followups",
        f"{RUN_KEY}:followup",
        {
            "conversation_id": SESSION_ID,
            "contact_name": CUSTOMER,
            "phone": PHONE,
            "due_at": now + timedelta(days=1),
            "template_name": "e2e_property_followup",
            "template_language": "en_US",
            "status": "pending",
            "notes": "E2E demo: confirm whether the customer wants a site visit.",
        },
    )
    await upsert(
        "search_sessions",
        f"{RUN_KEY}:recommendation",
        {
            "session_id": SESSION_ID,
            "name": f"{CUSTOMER} property recommendations",
            "city": "Bengaluru",
            "budget_max": 10000000,
            "matched_properties": [item["title"] for item in properties],
            "status": "active",
        },
    )
    await upsert(
        "public_form_leads",
        f"{RUN_KEY}:form",
        {
            "submission_id": RUN_KEY,
            "name": CUSTOMER,
            "phone": PHONE,
            "email": EMAIL,
            **lead_profile,
            "budget": "₹1 crore",
            "whatsapp_consent": "true",
            "status": "qualified",
            "source": "e2e_saved_demo",
        },
    )
    await upsert(
        "sales_leads",
        f"{RUN_KEY}:managed-lead",
        {
            "name": "Meera E2E Buyer",
            "phone": "919900001011",
            "email": "meera.e2e@example.com",
            "company": "Meera Investments",
            "property_purpose": "invest",
            "city": "Bengaluru",
            "localities": ["Sarjapur Road", "Whitefield"],
            "property_type": "Residential plot",
            "bedrooms": 0,
            "budget_min": 8000000,
            "budget_max": 12000000,
            "budget_currency": "INR",
            "timeline": "Within 6 months",
            "financing": "Self-funded",
            "query_reason": "Investment plot with road access in a gated layout.",
            "platform_name": "Pratap One manual lead",
            "form_name": "Sales desk qualification",
            "status": "qualified",
            "priority": "High",
            "score": 78,
            "assigned_to": salesperson_id,
            "assigned_name": "E2E Sales Advisor",
            "next_follow_up_at": now + timedelta(days=2),
            "follow_up_status": "scheduled",
            "follow_up_reason": "Share plot documents and arrange a consultation",
            "notes": "Second fully populated editable dummy lead.",
            "tags": ["E2E", "investor", "plot"],
            "source": "e2e_saved_demo",
        },
    )
    modules = {
        "marketing": ("E2E Bengaluru buyer campaign", "Active", "Campaigns", 0),
        "crm": (f"{CUSTOMER} — qualified lead", "Qualified", "All leads", 10000000),
        "wacrm": (f"{CUSTOMER} WhatsApp journey", "Open", "Conversations", 0),
        "recommendations": ("E2E 2 BHK shortlist", "Ready", "Results", 0),
        "visits": ("E2E Lakeview site visit", "Scheduled", "Calendar", 0),
        "pipeline": ("E2E Lakeview opportunity", "Qualified", "Kanban", 8200000),
        "bookings": ("E2E Lakeview draft booking", "Draft", "Booking list", 8200000),
        "finance": ("E2E token payment", "Recorded", "Collections", 100000),
        "work": ("Call E2E customer before visit", "Open", "My work", 0),
        "team": ("E2E Sales Advisor", "Active", "Salespeople", 0),
        "reports": ("E2E conversion test report", "Generated", "Sales", 8200000),
        "ai": ("E2E grounded property response", "Verified", "Insights", 0),
        "automation": ("E2E 5-hour follow-up flow", "Active", "Workflows", 0),
        "settings": ("E2E property enquiry template", "Approved", "Templates", 0),
        "partners": ("E2E Home Loan Partner", "Active", "Partners", 0),
        "legal": ("E2E title-document checklist", "Pending", "Due diligence", 0),
        "commissions": ("E2E advisor commission", "Projected", "Payouts", 82000),
    }
    for module, (title, status, view, value) in modules.items():
        await upsert(
            "pratap_one_records",
            f"{RUN_KEY}:module:{module}",
            {
                "module": module,
                "title": title,
                "meta": f"Saved full-CRM dummy data · customer {CUSTOMER} · property {property_row['title']}",
                "status": status,
                "owner": "E2E Sales Advisor",
                "view": view,
                "priority": "High" if module in {"crm", "visits", "work"} else "Normal",
                "due_date": (now + timedelta(days=1)).date().isoformat(),
                "value": value,
            },
        )
    return {"salesperson_id": salesperson_id, "lead_id": lead_id, "visit_id": visit_id, "booking_id": booking_id}


async def test_chatbot(properties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    await db.messages.delete_many({"session_id": SESSION_ID})
    prompts = [
        "Show me a 2 BHK property in Bengaluru below 1 crore near metro",
        "What documents should I verify before buying a property?",
    ]
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        await save_message(SESSION_ID, "user", prompt, channel="e2e_saved_demo")
        result = await graph.ainvoke({"session_id": SESSION_ID, "message": prompt, "image": None})
        answer = str(result.get("answer") or "")
        matches = [str(item.get("title")) for item in result.get("matches", [])]
        await save_message(
            SESSION_ID,
            "assistant",
            answer,
            channel="e2e_saved_demo",
            listings=matches,
            grounded=bool(result.get("grounded")),
            model=result.get("model"),
        )
        results.append({
            "prompt": prompt,
            "answer": answer,
            "matches": matches,
            "grounded": bool(result.get("grounded")),
            "model": result.get("model"),
        })
    if not any(properties[0]["title"] in item["matches"] for item in results):
        raise AssertionError("The saved E2E flat was not retrieved by the property-search flow")
    if not all(item["answer"].strip() for item in results):
        raise AssertionError("The chatbot returned an empty answer")
    return results


async def test_cancellation() -> dict[str, Any]:
    sent: list[str] = []

    async def capture_send(_phone: str, message: str) -> None:
        sent.append(message)

    cancelled = await _cancel_saved_booking(
        PHONE,
        "My plans changed, please cancel the appointment and site visit",
        send=capture_send,
    )
    if not cancelled:
        raise AssertionError("The chatbot did not understand the cancellation request")
    lead = await db.leads.find_one({"e2e_key": f"{RUN_KEY}:lead"})
    visit = await db.site_visits.find_one({"e2e_key": f"{RUN_KEY}:visit"})
    campaign = await db.outbound_campaigns.find_one({"e2e_key": f"{RUN_KEY}:campaign"})
    states = {
        "lead": lead.get("appointment_status") if lead else None,
        "visit": visit.get("status") if visit else None,
        "campaign": campaign.get("status") if campaign else None,
    }
    if states != {"lead": "cancelled", "visit": "cancelled", "campaign": "cancelled"}:
        raise AssertionError(f"Cancellation was not synchronized: {states}")
    await save_message(SESSION_ID, "user", "Please cancel my appointment", channel="e2e_saved_demo")
    await save_message(SESSION_ID, "assistant", sent[-1], channel="e2e_saved_demo")
    return {"understood": True, "states": states, "reply": sent[-1]}


async def save_summary(chat_results: list[dict[str, Any]], cancellation: dict[str, Any]) -> None:
    await upsert(
        "conversation_summaries",
        f"{RUN_KEY}:summary",
        {
            "conversation_id": SESSION_ID,
            "summary": "E2E customer searched for a Bengaluru 2 BHK near metro, reviewed grounded buying guidance, then cancelled the test appointment.",
            "requirement": "2 BHK in Bengaluru below ₹1 crore near metro",
            "budget": "₹1 crore",
            "sentiment": "Test journey completed",
            "nextAction": "No action—this is persistent E2E dummy data.",
            "latestCustomerMessage": "Please cancel my appointment",
            "message_count": len(chat_results) * 2 + 2,
            "model": "langgraph-rag-e2e",
        },
    )
    collections = [
        "properties", "leads", "sales_leads", "public_form_leads", "site_visits",
        "bookings", "outbound_campaigns", "crm_followups", "search_sessions",
        "pratap_one_records", "messages", "conversation_summaries", "crm_users",
    ]
    counts = {
        name: await db[name].count_documents(
            {"$or": [{"e2e_key": {"$regex": f"^{RUN_KEY}"}}, {"session_id": SESSION_ID}]}
        )
        for name in collections
    }
    await upsert(
        "e2e_test_runs",
        RUN_KEY,
        {
            "status": "passed",
            "session_id": SESSION_ID,
            "customer": CUSTOMER,
            "phone": PHONE,
            "collections": counts,
            "chat_results": chat_results,
            "cancellation": cancellation,
            "completed_at": datetime.now(timezone.utc),
        },
    )


async def main() -> None:
    await client.admin.command("ping")
    await initialize_database()
    properties = await seed_properties()
    ids = await seed_crm_journey(properties)
    chat_results = await test_chatbot(properties)
    cancellation = await test_cancellation()
    await save_summary(chat_results, cancellation)
    print(json.dumps({
        "status": "passed",
        "run_key": RUN_KEY,
        "session_id": SESSION_ID,
        "saved": True,
        "properties": [item["title"] for item in properties],
        "ids": ids,
        "chat": [{"matches": item["matches"], "grounded": item["grounded"], "model": item["model"]} for item in chat_results],
        "cancellation": cancellation["states"],
    }, indent=2, default=str))
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
