from datetime import datetime, timezone
from pymongo import AsyncMongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.server_api import ServerApi
from .config import get_settings

settings = get_settings()
client = AsyncMongoClient(settings.mongodb_uri, server_api=ServerApi("1"))
db = client[settings.mongodb_database]
properties = db.properties
messages = db.messages
knowledge_base = db.knowledge_base


async def initialize_database() -> None:
    await properties.create_index([("city", ASCENDING), ("price", ASCENDING)])
    await properties.create_index([("type", ASCENDING), ("bedrooms", ASCENDING)])
    await messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
    await messages.create_index("created_at", expireAfterSeconds=60 * 60 * 24 * 90)
    await db.webhook_events.create_index("message_id", unique=True)
    await db.webhook_events.create_index("created_at", expireAfterSeconds=60 * 60 * 24 * 7)
    await db.webhook_errors.create_index(
        "created_at", expireAfterSeconds=60 * 60 * 24 * 30
    )
    await db.leads.create_index([("phone", ASCENDING), ("property_title", ASCENDING)])
    await db.leads.create_index([("status", ASCENDING), ("updated_at", DESCENDING)])
    await db.customer_profiles.create_index("phone", unique=True)
    await db.booking_history.create_index(
        [("phone", ASCENDING), ("created_at", DESCENDING)]
    )
    await db.crm_users.create_index([("role", ASCENDING), ("active", ASCENDING)])
    await db.site_visits.create_index([("scheduled_at", ASCENDING), ("status", ASCENDING)])
    await db.site_visits.create_index([("owner_id", ASCENDING), ("scheduled_at", ASCENDING)])
    await db.outbound_campaigns.create_index("phone")
    await db.outbound_campaigns.create_index(
        [("status", ASCENDING), ("next_followup_at", ASCENDING)]
    )
    await db.public_form_leads.create_index([("phone", ASCENDING), ("created_at", DESCENDING)])
    await db.public_form_leads.create_index("submission_id", unique=True)
    await db.search_sessions.create_index("session_id", unique=True)
    await db.search_sessions.create_index(
        "updated_at", expireAfterSeconds=60 * 60 * 24 * 30
    )
    await db.conversation_turns.create_index(
        "received_at", expireAfterSeconds=60 * 60 * 24 * 7
    )
    await db.conversation_locks.create_index(
        "locked_until", expireAfterSeconds=0
    )
    await knowledge_base.create_index(
        [("title", TEXT), ("content", TEXT), ("keywords", TEXT), ("category", TEXT)],
        name="knowledge_text",
        weights={"title": 10, "keywords": 6, "category": 3, "content": 1},
    )
    await knowledge_base.create_index([("active", ASCENDING), ("category", ASCENDING)])


async def seed_knowledge_base() -> int:
    """Create a small starter KB without overwriting business-owned content."""
    documents = [
        {
            "slug": "property-buying-process-india",
            "title": "Property buying process in India",
            "category": "buying_guide",
            "keywords": ["buy", "purchase", "documents", "registration", "due diligence"],
            "content": (
                "A typical purchase includes requirement and budget discovery, property "
                "shortlisting, a site visit, title and approval due diligence by qualified "
                "professionals, price negotiation, agreement, payment, registration and handover."
            ),
            "source": "EstateAI operating guide",
            "active": True,
        },
        {
            "slug": "rental-process-india",
            "title": "Residential rental process in India",
            "category": "renting_guide",
            "keywords": ["rent", "lease", "deposit", "tenant", "agreement"],
            "content": (
                "Confirm monthly rent, security deposit, maintenance, notice period, included "
                "furnishings and utility responsibilities. Inspect the home and use a written, "
                "properly executed rental agreement after verifying the parties and property."
            ),
            "source": "EstateAI operating guide",
            "active": True,
        },
        {
            "slug": "listing-accuracy-policy",
            "title": "Listing and availability policy",
            "category": "policy",
            "keywords": ["available", "verified", "price", "listing", "accuracy"],
            "content": (
                "Listing price, availability and property facts must come from the current "
                "property database. Availability can change and should be reconfirmed with the "
                "assigned property expert before payment or travel."
            ),
            "source": "EstateAI policy",
            "active": True,
        },
        {
            "slug": "property-legal-due-diligence-india",
            "title": "Property legal due diligence checklist",
            "category": "legal_guide",
            "keywords": ["title deed", "encumbrance", "approval", "occupancy", "RERA", "tax receipt"],
            "content": (
                "Before a purchase, have an independent property lawyer verify ownership and "
                "the title chain, encumbrances, sanctioned plans and approvals, property-tax "
                "receipts, completion or occupancy documents where applicable, seller authority "
                "and the relevant RERA registration. Requirements vary, so confirm the checklist "
                "with local professionals before paying a token or signing."
            ),
            "source": "EstateAI due-diligence guide",
            "active": True,
        },
        {
            "slug": "home-loan-planning-india",
            "title": "Home loan planning guide",
            "category": "finance_guide",
            "keywords": ["home loan", "mortgage", "EMI", "down payment", "interest", "eligibility"],
            "content": (
                "Plan a home loan by comparing the down payment, EMI affordability, interest-rate "
                "structure, tenure, processing and insurance costs, prepayment terms and total "
                "interest—not only the advertised rate. Lenders determine eligibility and property "
                "acceptance; obtain current written offers and independent financial advice."
            ),
            "source": "EstateAI finance guide",
            "active": True,
        },
        {
            "slug": "property-site-visit-checklist",
            "title": "Property site visit checklist",
            "category": "inspection_guide",
            "keywords": ["site visit", "inspection", "water", "ventilation", "noise", "condition"],
            "content": (
                "During a site visit, inspect daylight, ventilation, water pressure, seepage, "
                "electrical points, room measurements, common areas, lifts, parking and access. "
                "Check daytime and evening noise, nearby services and actual travel time. Record "
                "defects and reconcile every claim with documents and qualified inspections."
            ),
            "source": "EstateAI inspection guide",
            "active": True,
        },
        {
            "slug": "property-investment-comparison",
            "title": "Property investment comparison framework",
            "category": "investment_guide",
            "keywords": ["investment", "rental yield", "ROI", "appreciation", "vacancy", "resale"],
            "content": (
                "Compare investment properties using total acquisition cost, realistic rent, "
                "vacancy, maintenance, taxes, financing cost, net rental yield, tenant demand, "
                "liquidity and resale risk. Appreciation is uncertain, so test conservative cash "
                "flows and verify local market evidence rather than relying on projections."
            ),
            "source": "EstateAI investment guide",
            "active": True,
        },
        {
            "slug": "property-selling-process-india",
            "title": "Property selling process",
            "category": "selling_guide",
            "keywords": ["sell", "selling", "resale", "valuation", "buyer", "offer"],
            "content": (
                "A sale usually includes document readiness, evidence-based pricing, listing and "
                "buyer qualification, visits, offer comparison, negotiation, agreement, payment "
                "verification, registration and handover. Ask legal and tax professionals to "
                "review the transaction and current obligations before execution."
            ),
            "source": "EstateAI selling guide",
            "active": True,
        },
        {
            "slug": "property-valuation-method",
            "title": "Property valuation methodology",
            "category": "valuation_guide",
            "keywords": ["valuation", "market value", "price estimate", "comparable", "rate per sqft"],
            "content": (
                "A defensible valuation compares recent similar transactions and adjusts for exact "
                "location, usable area, property age and condition, floor, view, parking, amenities, "
                "tenure and market liquidity. An online estimate is indicative; use current local "
                "comparables and a qualified valuer when the decision is material."
            ),
            "source": "EstateAI valuation guide",
            "active": True,
        },
    ]
    inserted = 0
    now = datetime.now(timezone.utc)
    for document in documents:
        result = await knowledge_base.update_one(
            {"slug": document["slug"]},
            {"$setOnInsert": {**document, "created_at": now, "updated_at": now}},
            upsert=True,
        )
        inserted += int(result.upserted_id is not None)
    return inserted


async def save_message(session_id: str, role: str, content: str, **extra) -> None:
    await messages.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc),
        **extra,
    })


async def get_history(session_id: str, limit: int) -> list[dict]:
    cursor = messages.find(
        {"session_id": session_id},
        {"_id": 0, "role": 1, "content": 1, "images": 1},
    ).sort("created_at", DESCENDING).limit(limit)
    rows = [row async for row in cursor]
    return list(reversed(rows))


async def seed_demo_properties() -> int:
    demo = [
        {"title": "Skyline Residences", "city": "Mumbai", "locality": "Worli",
         "type": "apartment", "bedrooms": 3, "bathrooms": 3, "area_sqft": 1850,
         "price": 72500000, "currency": "INR", "purpose": "buy",
         "amenities": ["Sea view", "Pool", "Gym", "24/7 security"],
         "description": "Premium sea-facing home with a private balcony.",
         "images": ["https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?auto=format&fit=crop&w=1200&q=80"]},
        {"title": "Green Courtyard Villa", "city": "Bengaluru", "locality": "Whitefield",
         "type": "villa", "bedrooms": 4, "bathrooms": 4, "area_sqft": 3200,
         "price": 48000000, "currency": "INR", "purpose": "buy",
         "amenities": ["Private garden", "Clubhouse", "Solar power", "Garage"],
         "description": "Quiet gated-community villa close to the IT corridor.",
         "images": ["https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?auto=format&fit=crop&w=1200&q=80"]},
        {"title": "Urban Nest", "city": "Gurugram", "locality": "Golf Course Road",
         "type": "apartment", "bedrooms": 2, "bathrooms": 2, "area_sqft": 1250,
         "price": 85000, "currency": "INR", "purpose": "rent",
         "amenities": ["Furnished", "Metro nearby", "Gym", "Power backup"],
         "description": "Fully furnished modern apartment, ready to move in.",
         "images": ["https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=1200&q=80"]},
        {"title": "Palm Breeze Plot", "city": "Goa", "locality": "Assagao",
         "type": "plot", "bedrooms": 0, "bathrooms": 0, "area_sqft": 5400,
         "price": 39000000, "currency": "INR", "purpose": "buy",
         "amenities": ["Clear title", "Road access", "Green zone"],
         "description": "Build-ready plot in a peaceful, sought-after village.",
         "images": ["https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80"]},
        {"title": "Aravalli Heights", "city": "Gurugram", "locality": "Sector 54",
         "type": "apartment", "bedrooms": 3, "bathrooms": 3, "area_sqft": 2100,
         "price": 34000000, "currency": "INR", "purpose": "buy",
         "amenities": ["Golf course view", "Pool", "Concierge", "Gym"],
         "description": "High-floor residence with open green views and clubhouse access.",
         "images": [
             "https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3?auto=format&fit=crop&w=1200&q=80",
             "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?auto=format&fit=crop&w=1200&q=80"]},
        {"title": "Indiranagar Studio", "city": "Bengaluru", "locality": "Indiranagar",
         "type": "apartment", "bedrooms": 1, "bathrooms": 1, "area_sqft": 650,
         "price": 42000, "currency": "INR", "purpose": "rent",
         "amenities": ["Furnished", "Pet friendly", "Metro nearby", "Balcony"],
         "description": "Bright furnished studio near cafés and the metro.",
         "images": [
             "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?auto=format&fit=crop&w=1200&q=80",
             "https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?auto=format&fit=crop&w=1200&q=80"]},
        {"title": "Jubilee Garden Home", "city": "Hyderabad", "locality": "Jubilee Hills",
         "type": "villa", "bedrooms": 5, "bathrooms": 5, "area_sqft": 5100,
         "price": 115000000, "currency": "INR", "purpose": "buy",
         "amenities": ["Private pool", "Home theatre", "Garden", "Staff quarters"],
         "description": "Contemporary independent home on a quiet tree-lined street.",
         "images": [
             "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1200&q=80",
             "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?auto=format&fit=crop&w=1200&q=80"]},
        {"title": "Baner Work-Life Residence", "city": "Pune", "locality": "Baner",
         "type": "apartment", "bedrooms": 2, "bathrooms": 2, "area_sqft": 1080,
         "price": 32000, "currency": "INR", "purpose": "rent",
         "amenities": ["Coworking lounge", "Pool", "Parking", "Power backup"],
         "description": "Well-connected gated apartment suited to working professionals.",
         "images": [
             "https://images.unsplash.com/photo-1493809842364-78817add7ffb?auto=format&fit=crop&w=1200&q=80",
             "https://images.unsplash.com/photo-1560185008-b033106af5c3?auto=format&fit=crop&w=1200&q=80"]},
    ]

    # Generate a broad, deterministic demo catalogue. These records deliberately
    # cover different budgets and intents so the LangGraph search can demonstrate
    # meaningful filtering, comparisons, and automatic image recommendations.
    markets = [
        ("Mumbai", "Powai", 19.1197, 72.9050, 23000000),
        ("Mumbai", "Andheri West", 19.1364, 72.8296, 26000000),
        ("Bengaluru", "Koramangala", 12.9352, 77.6245, 17000000),
        ("Bengaluru", "Sarjapur Road", 12.9008, 77.6913, 12500000),
        ("Gurugram", "Sector 67", 28.4053, 77.0570, 19000000),
        ("Gurugram", "Dwarka Expressway", 28.4464, 76.9956, 16500000),
        ("Pune", "Kharadi", 18.5515, 73.9348, 11000000),
        ("Pune", "Hinjawadi", 18.5913, 73.7389, 8500000),
        ("Hyderabad", "Gachibowli", 17.4401, 78.3489, 14500000),
        ("Hyderabad", "Kokapet", 17.3948, 78.3377, 16000000),
        ("Chennai", "OMR", 12.9000, 80.2279, 10500000),
        ("Chennai", "Anna Nagar", 13.0850, 80.2101, 18500000),
        ("Noida", "Sector 150", 28.4484, 77.4810, 13000000),
        ("Noida", "Sector 75", 28.5708, 77.3818, 11500000),
    ]
    project_names = [
        "Aster", "Celeste", "Elara", "Ivy", "Meridian", "Opal", "Orchid",
        "Serene", "Solstice", "Verde", "Willow", "Zenith",
    ]
    image_sets = [
        [
            "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?auto=format&fit=crop&w=1200&q=80",
            "https://images.unsplash.com/photo-1560185007-c5ca9d2c014d?auto=format&fit=crop&w=1200&q=80",
        ],
        [
            "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=1200&q=80",
            "https://images.unsplash.com/photo-1600566753051-f0b89df2dd90?auto=format&fit=crop&w=1200&q=80",
        ],
        [
            "https://images.unsplash.com/photo-1600585152915-d208bec867a1?auto=format&fit=crop&w=1200&q=80",
            "https://images.unsplash.com/photo-1613490493576-7fde63acd811?auto=format&fit=crop&w=1200&q=80",
        ],
        [
            "https://images.unsplash.com/photo-1600607688969-a5bfcd646154?auto=format&fit=crop&w=1200&q=80",
            "https://images.unsplash.com/photo-1600210491892-03d54c0aaf87?auto=format&fit=crop&w=1200&q=80",
        ],
        [
            "https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3?auto=format&fit=crop&w=1200&q=80",
            "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?auto=format&fit=crop&w=1200&q=80",
        ],
        [
            "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1200&q=80",
            "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?auto=format&fit=crop&w=1200&q=80",
        ],
    ]
    amenity_sets = [
        ["Swimming pool", "Gym", "Clubhouse", "24/7 security", "Parking"],
        ["Metro nearby", "Power backup", "Children's play area", "CCTV", "Garden"],
        ["Coworking lounge", "Rooftop deck", "EV charging", "Concierge", "Gym"],
        ["Pet friendly", "Jogging track", "Sports court", "Rainwater harvesting", "Pool"],
    ]
    agents = [
        ("Riya Mehta", "+91 90000 10001", "riya@example.com"),
        ("Arjun Kapoor", "+91 90000 10002", "arjun@example.com"),
        ("Neha Sharma", "+91 90000 10003", "neha@example.com"),
        ("Kabir Singh", "+91 90000 10004", "kabir@example.com"),
    ]

    for index in range(92):
        city, locality, latitude, longitude, base_price = markets[index % len(markets)]
        bedrooms = (index % 4) + 1
        purpose = "rent" if index % 5 in {1, 4} else "buy"
        property_type = "villa" if bedrooms == 4 and index % 3 == 0 else "apartment"
        if purpose == "rent":
            # Keep broad rental coverage, including realistic sub-₹40k options
            # across every generated market for common chatbot demo searches.
            price = 16000 + (index % 5) * 3500 + bedrooms * 4500
        else:
            price = int(base_price * (0.68 + bedrooms * 0.17 + (index % 4) * 0.06))
        area = 520 + bedrooms * 410 + (index % 6) * 85
        agent_name, agent_phone, agent_email = agents[index % len(agents)]
        demo.append({
            "title": f"{project_names[index % len(project_names)]} {locality} {index + 101}",
            "city": city,
            "locality": locality,
            "type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": max(1, bedrooms),
            "balconies": 1 + index % 3,
            "area_sqft": area,
            "carpet_area_sqft": int(area * 0.78),
            "price": price,
            "currency": "INR",
            "purpose": purpose,
            "price_negotiable": index % 3 == 0,
            "maintenance_monthly": 2500 + bedrooms * 1400,
            "furnishing": ["unfurnished", "semi-furnished", "fully furnished"][index % 3],
            "possession_status": ["ready to move", "under construction"][index % 2],
            "available_from": "2026-08-15" if purpose == "rent" else None,
            "floor": index % 22 + 1,
            "total_floors": 24,
            "age_years": index % 8,
            "facing": ["East", "West", "North", "North-East"][index % 4],
            "parking_spaces": 1 if bedrooms < 3 else 2,
            "amenities": amenity_sets[index % len(amenity_sets)],
            "description": (
                f"A thoughtfully planned {bedrooms} BHK {property_type} in {locality}, "
                f"{city}, with abundant daylight, efficient rooms and convenient access "
                "to daily essentials. Demo listing for EstateAI recommendations."
            ),
            "nearby": ["School", "Hospital", "Supermarket", "Public transport"],
            "coordinates": {"latitude": latitude + index * 0.0001,
                            "longitude": longitude + index * 0.0001},
            "rera_registered": index % 4 != 0,
            "featured": index % 7 == 0,
            "verified": True,
            "agent": {"name": agent_name, "phone": agent_phone, "email": agent_email},
            "images": image_sets[index % len(image_sets)],
            "created_at": datetime.now(timezone.utc),
        })
    inserted = 0
    for item in demo:
        result = await properties.update_one(
            {"title": item["title"], "city": item["city"]},
            {"$setOnInsert": item},
            upsert=True,
        )
        inserted += int(result.upserted_id is not None)
    return inserted
