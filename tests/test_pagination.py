import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import agent, whatsapp
from app.whatsapp import delivery_image_url


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, *_):
        self.rows.sort(key=lambda row: row["price"])
        return self

    def skip(self, count):
        self.rows = self.rows[count:]
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def __aiter__(self):
        self._iterator = iter(self.rows)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeProperties:
    def __init__(self):
        self.queries = []
        self.rows = [
            {"title": f"{city}-{index}", "city": city, "price": index, "images": ["https://img"]}
            for city in ("Bengaluru", "Pune", "Mumbai")
            for index in range(1, 9)
        ]

    def _filter(self, query):
        city_rule = query.get("city")
        if not city_rule:
            return self.rows
        pattern = city_rule["$regex"]
        return [row for row in self.rows if re.match(pattern, row["city"], re.I)]

    async def count_documents(self, query):
        self.queries.append(query)
        return len(self._filter(query))

    def find(self, query, _projection):
        self.queries.append(query)
        return FakeCursor(self._filter(query))


class FakePropertyContextStore:
    def __init__(self):
        self.rows = [
            {
                "title": "Aravalli Heights", "city": "Gurugram",
                "price": 34_000_000, "parking_spaces": 2,
                "area_sqft": 1900, "images": ["one", "two"],
            },
            {
                "title": "Aster Gachibowli 137", "city": "Hyderabad",
                "price": 27_500, "purpose": "rent", "images": ["three"],
            },
        ]

    def find(self, _query, projection):
        return FakeCursor([
            {key: row[key] for key in projection if key != "_id" and key in row}
            for row in self.rows
        ])

    async def find_one(self, query, _projection):
        return next((row for row in self.rows if row["title"] == query["title"]), None)


class FakeSearchSessions:
    def __init__(self):
        self.row = None

    async def find_one(self, query):
        return self.row if self.row and self.row["session_id"] == query["session_id"] else None

    async def update_one(self, query, update, upsert=False):
        self.row = {**(self.row or {}), **update["$set"]}


class PaginationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ai_classifies_natural_interest_and_preserves_canonical_title(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=(
                '{"is_interested":true,'
                '"selected_property":"Ivy Sarjapur Road 104"}'
            ))
        )])
        with patch.object(
            agent.llm.chat.completions, "create", AsyncMock(return_value=response)
        ):
            decision = await agent.classify_property_response(
                "Ivy Sarjapur Road 104 this one seems perfect for us",
                ["Opal Sarjapur Road 118", "Ivy Sarjapur Road 104"],
            )
        self.assertTrue(decision["is_interested"])
        self.assertEqual(decision["selected_property"], "Ivy Sarjapur Road 104")

    async def test_ai_does_not_treat_a_new_city_search_as_interest(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=(
                '{"is_interested":false,"selected_property":null}'
            ))
        )])
        with patch.object(
            agent.llm.chat.completions, "create", AsyncMock(return_value=response)
        ):
            decision = await agent.classify_property_response(
                "Show properties in Pune",
                ["Opal Sarjapur Road 118", "Ivy Sarjapur Road 104"],
            )
        self.assertFalse(decision["is_interested"])
        self.assertIsNone(decision["selected_property"])

    async def test_selected_property_question_is_not_forced_into_booking(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=(
                '{"action":"question","wants_images":true,'
                '"answer":"I can show the available gallery."}'
            ))
        )])
        with patch.object(
            agent, "retrieve_knowledge", AsyncMock(return_value={"knowledge": []})
        ), patch.object(
            agent.llm.chat.completions, "create", AsyncMock(return_value=response)
        ):
            decision = await agent.advise_on_selected_property(
                "Do you have more photos of this one?",
                {"title": "Aster Koramangala 173", "images": ["one", "two"]},
            )
        self.assertEqual(decision["action"], "question")
        self.assertTrue(decision["wants_images"])

    async def test_booking_is_routed_only_after_customer_accepts(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=(
                '{"action":"book","wants_images":false,"answer":""}'
            ))
        )])
        with patch.object(
            agent, "retrieve_knowledge", AsyncMock(return_value={"knowledge": []})
        ), patch.object(
            agent.llm.chat.completions, "create", AsyncMock(return_value=response)
        ):
            decision = await agent.advise_on_selected_property(
                "Yes, please arrange a call", {"title": "Aster Koramangala 173"}
            )
        self.assertEqual(decision["action"], "book")

    async def test_existing_booking_can_be_cancelled_in_natural_language(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"cancel":true}')
        )])
        with patch.object(
            agent.llm.chat.completions, "create", AsyncMock(return_value=response)
        ):
            cancelled = await agent.classify_booking_cancellation(
                "I changed my mind, please withdraw tomorrow's appointment",
                {"property_title": "Aster Koramangala 173", "status": "confirmed"},
            )
        self.assertTrue(cancelled)

    async def test_call_off_site_visit_is_cancelled_without_llm(self):
        completion = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        with patch.object(agent.llm.chat.completions, "create", completion):
            cancelled = await agent.classify_booking_cancellation(
                "call off the site visit",
                {"property_title": "Aravalli Heights", "status": "confirmed"},
            )
        self.assertTrue(cancelled)
        completion.assert_not_called()

    async def test_unrelated_no_does_not_cancel_existing_booking(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"cancel":false}')
        )])
        with patch.object(
            agent.llm.chat.completions, "create", AsyncMock(return_value=response)
        ):
            cancelled = await agent.classify_booking_cancellation(
                "No, show me the kitchen photos",
                {"property_title": "Aster Koramangala 173", "status": "confirmed"},
            )
        self.assertFalse(cancelled)

    def test_unsplash_images_use_stable_delivery_proxy(self):
        with patch.object(whatsapp.settings, "app_url", "https://estate.example"):
            proxied = delivery_image_url(
                "https://images.unsplash.com/photo-test?auto=format&fit=crop&w=1200"
            )
        self.assertTrue(proxied.startswith("https://"))
        self.assertIn("/api/property-image?url=", proxied)
        self.assertNotIn("&fit=", proxied)

    def test_greeting_is_detected_without_matching_property_search(self):
        self.assertTrue(agent._is_greeting("Hii!"))
        self.assertFalse(agent._is_greeting("I want to buy property"))

    def test_fresh_property_request_does_not_reuse_an_old_city(self):
        self.assertTrue(
            agent._starts_locationless_search(
                "I want to buy property", {"intent": "property_search", "purpose": "buy"}
            )
        )
        self.assertFalse(
            agent._starts_locationless_search(
                "I want property in Pune", {"intent": "property_search", "city": "Pune"}
            )
        )

    def test_understands_city_typo_and_multiple_cities(self):
        self.assertEqual(agent._heuristic_intent("properties in bengulore")["city"], "Bengaluru")
        self.assertEqual(
            set(agent._heuristic_intent("show Pune and Mumbai properties")["cities"]),
            {"Pune", "Mumbai"},
        )

    def test_property_name_is_resolved_inside_a_natural_question(self):
        titles = ["Ivy Sarjapur Road 104", "Aster Koramangala 173"]
        self.assertEqual(
            agent._best_referenced_title(
                "Does Aster Koramangala 173 include two parking spaces?", titles
            ),
            "Aster Koramangala 173",
        )
        self.assertEqual(
            agent._referenced_titles(
                "Compare Aster Koramangala 173 and Ivy Sarjapur Road 104", titles
            ),
            ["Ivy Sarjapur Road 104", "Aster Koramangala 173"],
        )
        self.assertEqual(
            agent._best_referenced_title("show more photos of aster 173", titles),
            "Aster Koramangala 173",
        )

    def test_property_fact_words_are_not_treated_as_title_tokens(self):
        self.assertEqual(
            agent._reference_tokens(
                "What is the price, area, parking, amenities and location of Aravalli Heights?"
            ),
            ["aravalli", "heights"],
        )

    async def test_named_property_and_comparison_resolve_from_database(self):
        pending = SimpleNamespace(find_one=AsyncMock(return_value=None))
        sessions = FakeSearchSessions()
        store = FakePropertyContextStore()
        with patch.object(
            agent, "db", SimpleNamespace(pending_leads=pending, search_sessions=sessions)
        ), patch.object(
            agent, "properties", store
        ):
            detail = await agent.resolve_property_context({
                "session_id": "details", "history": [], "matches": [],
                "intent": {"intent": "general"},
                "message": "What is the price and parking of Aravalli Heights?",
            })
            comparison = await agent.resolve_property_context({
                "session_id": "comparison", "history": [], "matches": [],
                "intent": {"intent": "general"},
                "message": "Compare Aravalli Heights and Aster Gachibowli 137",
            })
        self.assertEqual(detail["matches"][0]["title"], "Aravalli Heights")
        self.assertEqual(comparison["intent"]["intent"], "compare")
        self.assertEqual(len(comparison["matches"]), 2)

    async def test_pronoun_followup_uses_named_property_from_history(self):
        pending = SimpleNamespace(find_one=AsyncMock(return_value=None))
        sessions = FakeSearchSessions()
        with patch.object(
            agent, "db", SimpleNamespace(pending_leads=pending, search_sessions=sessions)
        ), patch.object(
            agent, "properties", FakePropertyContextStore()
        ):
            result = await agent.resolve_property_context({
                "session_id": "follow-up", "matches": [],
                "intent": {"intent": "general"},
                "message": "Show its photos",
                "history": [{"role": "user", "content": "Tell me about Aravalli Heights"}],
            })
        self.assertEqual(result["matches"][0]["title"], "Aravalli Heights")
        self.assertTrue(result["page_info"]["focused_property"])

    async def test_selected_property_survives_expired_chat_history(self):
        pending = SimpleNamespace(find_one=AsyncMock(return_value=None))
        sessions = FakeSearchSessions()
        store = FakePropertyContextStore()
        database = SimpleNamespace(pending_leads=pending, search_sessions=sessions)
        with patch.object(agent, "db", database), patch.object(agent, "properties", store):
            selected = await agent.resolve_property_context({
                "session_id": "durable-follow-up", "history": [], "matches": [],
                "intent": {"intent": "general"},
                "message": "Tell me about Aravalli Heights",
            })
            follow_up = await agent.resolve_property_context({
                "session_id": "durable-follow-up", "history": [], "matches": [],
                "intent": {"intent": "general"}, "message": "pics bhejo",
            })
        self.assertEqual(selected["matches"][0]["title"], "Aravalli Heights")
        self.assertEqual(follow_up["matches"][0]["title"], "Aravalli Heights")

    def test_hinglish_property_discovery_and_rental_education_are_distinct(self):
        discovery = agent._heuristic_intent("property chahiye")
        education = agent._heuristic_intent("How do I calculate rental yield?")
        self.assertEqual(discovery["intent"], "property_search")
        self.assertNotIn("purpose", education)

    def test_seller_request_is_not_treated_as_buyer_search(self):
        for message in (
            "I want to sell my property",
            "Can you list my apartment?",
            "Mera ghar bechna hai",
        ):
            with self.subTest(message=message):
                intent = agent._heuristic_intent(message)
                self.assertEqual(intent["intent"], "seller_intake")
                self.assertEqual(intent["purpose"], "sell")

    async def test_seller_and_listing_rejection_responses_are_contextual(self):
        seller = await agent.compose_answer({
            "message": "I want to sell my property",
            "intent": {"intent": "seller_intake", "purpose": "sell"},
            "matches": [], "knowledge": [], "history": [], "page_info": {},
        })
        rejection = await agent.compose_answer({
            "message": "No", "intent": {"intent": "listing_rejection"},
            "matches": [], "knowledge": [], "history": [], "page_info": {},
        })
        self.assertIn("sell", seller["answer"].lower())
        self.assertIn("city and locality", seller["answer"].lower())
        self.assertIn("what should i change", rejection["answer"].lower())

    async def test_misclassified_detail_search_keeps_selected_property(self):
        sessions = FakeSearchSessions()
        sessions.row = {
            "session_id": "keep-selected", "selected_property": "Aravalli Heights"
        }
        with patch.object(agent, "properties", FakeProperties()), patch.object(
            agent, "db", SimpleNamespace(search_sessions=sessions)
        ):
            result = await agent.search_inventory({
                "session_id": "keep-selected", "message": "pics bhejo",
                "intent": {"intent": "property_search", "wants_images": True},
            })
        self.assertTrue(result["page_info"]["requires_city"])
        self.assertEqual(sessions.row["selected_property"], "Aravalli Heights")

    async def test_explicit_new_city_search_clears_selected_property(self):
        sessions = FakeSearchSessions()
        sessions.row = {
            "session_id": "clear-selected", "selected_property": "Aravalli Heights"
        }
        with patch.object(agent, "properties", FakeProperties()), patch.object(
            agent, "db", SimpleNamespace(search_sessions=sessions)
        ):
            await agent.search_inventory({
                "session_id": "clear-selected", "message": "show properties in Pune",
                "intent": {"intent": "property_search", "city": "Pune"},
            })
        self.assertIsNone(sessions.row["selected_property"])

    def test_count_question_is_detected_as_database_aggregation(self):
        intent = agent._heuristic_intent("How many properties are in Bengaluru?")
        self.assertEqual(intent["city"], "Bengaluru")
        self.assertEqual(intent["_aggregate"], "count")

    def test_common_real_estate_filters_work_without_llm(self):
        intent = agent._heuristic_intent(
            "Show ready to move semi furnished 3 BHK villas in Pune under 1800 sqft "
            "with 2 bathrooms, 2 car parking, swimming pool, gym and east facing"
        )
        self.assertEqual(intent["city"], "Pune")
        self.assertEqual(intent["property_type"], "villa")
        self.assertEqual(intent["min_bedrooms"], 3)
        self.assertEqual(intent["min_bathrooms"], 2)
        self.assertEqual(intent["max_area_sqft"], 1800)
        self.assertNotIn("max_price", intent)
        self.assertEqual(intent["furnishing"], "semi-furnished")
        self.assertEqual(intent["possession_status"], "ready to move")
        self.assertEqual(intent["facing"], "East")
        self.assertEqual(intent["min_parking"], 2)
        self.assertEqual(intent["amenities"], ["Swimming pool", "Gym"])

    def test_all_stored_property_fact_categories_have_grounded_answers(self):
        listing = {
            "title": "Test Residence", "price": 12_500_000, "currency": "INR",
            "purpose": "buy", "maintenance_monthly": 4500, "parking_spaces": 2,
            "area_sqft": 1450, "carpet_area_sqft": 1100, "bedrooms": 3,
            "bathrooms": 2, "balconies": 2, "furnishing": "semi-furnished",
            "floor": 8, "total_floors": 20, "facing": "East",
            "rera_registered": True, "verified": True,
            "possession_status": "ready to move", "available_from": "2026-08-15",
            "age_years": 3, "price_negotiable": False,
            "amenities": ["Gym", "Pool"], "nearby": ["Metro", "Hospital"],
            "locality": "Baner", "city": "Pune",
            "agent": {"name": "Riya"}, "images": ["one", "two"],
        }
        cases = {
            "What is the price?": "₹12,500,000",
            "What is monthly maintenance?": "₹4,500",
            "Does it have parking?": "Parking spaces: 2",
            "What is the carpet area?": "1,100 sq ft",
            "How many bedrooms and bathrooms?": "Bathrooms: 2",
            "How many balconies?": "Balconies: 2",
            "Is it furnished?": "semi-furnished",
            "Which floor?": "8 of 20",
            "Which facing?": "East",
            "Is it RERA registered and verified?": "RERA registered: Yes",
            "Is it ready to move and available?": "ready to move",
            "How old is it?": "3 years",
            "Is the price negotiable?": "Price negotiable: No",
            "What amenities are there?": "Gym, Pool",
            "What is nearby?": "Metro, Hospital",
            "Where is it located?": "Baner, Pune",
            "Who is the agent contact?": "Agent: Riya",
            "Show photos": "2 images",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertIn(expected, agent._database_property_answer(question, listing))

    def test_hinglish_property_followups_use_database_facts(self):
        listing = {
            "title": "Aravalli Heights", "price": 34_000_000,
            "currency": "INR", "parking_spaces": 2, "images": ["one", "two"],
        }
        self.assertIn("34,000,000", agent._database_property_answer("kitne ka hai?", listing))
        self.assertIn("2 images", agent._database_property_answer("pics bhejo", listing))
        self.assertIn("Parking spaces: 2", agent._database_property_answer(
            "gaadi kaha park hogi?", listing
        ))
        metro_answer = agent._database_property_answer(
            "nearest metro kitna door hai?",
            {**listing, "nearby_metro": {"status": "coordinates_missing"}},
        )
        self.assertIn("Nearest metro", metro_answer)
        self.assertNotIn("Price:", metro_answer)

    async def test_count_question_returns_total_without_property_gallery(self):
        sessions = FakeSearchSessions()
        with patch.object(agent, "properties", FakeProperties()), patch.object(
            agent, "db", SimpleNamespace(search_sessions=sessions)
        ):
            result = await agent.search_inventory({
                "session_id": "count-one",
                "intent": {
                    "intent": "property_search", "city": "Bengaluru",
                    "_aggregate": "count",
                },
            })
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["page_info"]["total"], 8)
        self.assertTrue(result["page_info"]["count_query"])

    async def test_amenity_filters_use_valid_mongodb_and_clauses(self):
        sessions = FakeSearchSessions()
        store = FakeProperties()
        with patch.object(agent, "properties", store), patch.object(
            agent, "db", SimpleNamespace(search_sessions=sessions)
        ):
            await agent.search_inventory({
                "session_id": "amenities",
                "intent": {
                    "intent": "property_search", "city": "Pune",
                    "amenities": ["Gym", "Swimming pool"],
                },
            })
        query = next(query for query in store.queries if "$and" in query)
        self.assertNotIn("$all", str(query))
        self.assertEqual(len(query["$and"]), 2)
        self.assertEqual(query["$and"][0]["amenities"]["$regex"], "Gym")

    async def test_ambiguous_intent_falls_back_when_free_model_quota_is_exhausted(self):
        with patch.object(
            agent, "get_history", AsyncMock(return_value=[])
        ), patch.object(
            agent.llm.chat.completions, "create",
            AsyncMock(side_effect=RuntimeError("429 free quota exhausted")),
        ):
            result = await agent.understand({
                "session_id": "fallback", "message": "I am confused, please guide me",
                "history": [],
            })
        self.assertEqual(result["intent"]["intent"], "general")
        self.assertEqual(result["model"], "deterministic-fallback")

    async def test_recognizable_database_question_skips_llm_understanding(self):
        create = AsyncMock(return_value=SimpleNamespace(
            model="test-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"intent":"property_search","city":"Pune"}'))],
        ))
        with patch.object(agent.llm.chat.completions, "create", create):
            result = await agent.understand({
                "session_id": "fast-count", "message": "How many properties are in Pune?",
                "history": [],
            })
        create.assert_not_awaited()
        self.assertEqual(result["intent"]["_aggregate"], "count")
        self.assertEqual(result["model"], "deterministic")

    async def test_llm_city_correction_excludes_rejected_city(self):
        completion = SimpleNamespace(
            model="test-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content=(
                '{"intent":"property_search","city":"Bengaluru",'
                '"excluded_cities":["Mumbai"]}'
            )))],
        )
        with patch.object(agent, "_create_completion", AsyncMock(return_value=completion)):
            result = await agent.understand({
                "session_id": "city-correction",
                "message": "No, I do not want Mumbai. Show me Bangalore instead",
                "history": [
                    {"role": "user", "content": "Mumbai"},
                    {"role": "assistant", "content": "Here are Mumbai properties"},
                ],
            })
        self.assertEqual(result["intent"].get("city"), "Bengaluru")
        self.assertNotIn("Mumbai", result["intent"].get("cities", []))
        self.assertEqual(result["intent"]["excluded_cities"], ["Mumbai"])

    async def test_partial_unique_property_name_is_resolved(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=(
                '{"is_interested":true,"selected_property":"Elara","is_question":false}'
            )))],
        )
        with patch.object(agent, "_create_completion", AsyncMock(return_value=completion)):
            result = await agent.classify_property_response(
                "I liked Elara",
                ["Elara Koramangala 187", "Solstice Koramangala 145"],
            )
        self.assertTrue(result["is_interested"])
        self.assertEqual(result["selected_property"], "Elara Koramangala 187")

    async def test_exact_property_selection_skips_llm(self):
        completion = AsyncMock()
        with patch.object(agent, "_create_completion", completion):
            result = await agent.classify_property_response(
                "I like Elara Koramangala 187",
                ["Elara Koramangala 187", "Solstice Koramangala 145"],
            )
        completion.assert_not_awaited()
        self.assertTrue(result["is_interested"])
        self.assertEqual(result["selected_property"], "Elara Koramangala 187")

    async def test_standard_schedule_skips_llm(self):
        completion = AsyncMock()
        now = agent.datetime(2026, 8, 12, 10, 0, tzinfo=agent.timezone.utc)
        with patch.object(agent, "_create_completion", completion):
            slot, has_date, has_time = await agent.extract_schedule(
                "15 Aug at 4 PM", now=now
            )
        completion.assert_not_awaited()
        self.assertTrue(has_date)
        self.assertTrue(has_time)
        self.assertIsNotNone(slot)

    async def test_plain_booking_request_skips_type_llm(self):
        completion = AsyncMock()
        with patch.object(agent, "_create_completion", completion):
            booking_type = await agent.classify_booking_type("yes book it")
        completion.assert_not_awaited()
        self.assertIsNone(booking_type)

    async def test_advisor_reply_rejects_internal_reasoning_leak(self):
        leaked = """Here's a thinking process:
1. **Analyze User Input:**
- **Required Message:** What email should I use?
- **Verified Context:** awaiting_details"""
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=leaked)
        )])
        with patch.object(agent, "_create_completion", AsyncMock(return_value=completion)):
            reply = await agent.generate_advisor_reply(
                "What email should I use to send the confirmation?", "hii"
            )
        self.assertEqual(reply, "What email should I use to send the confirmation?")

    async def test_advisor_reply_accepts_short_natural_acknowledgement(self):
        natural = "Got it, Devesh 👍 What email should I use for your confirmation?"
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=natural)
        )])
        with patch.object(agent, "_create_completion", AsyncMock(return_value=completion)):
            reply = await agent.generate_advisor_reply(
                "Ask for the confirmation email.", "My name is Devesh"
            )
        self.assertEqual(reply, natural)

    async def test_advisor_reply_never_falls_back_to_internal_instruction(self):
        with patch.object(
            agent, "_create_completion", AsyncMock(side_effect=RuntimeError("rate limited"))
        ):
            reply = await agent.generate_advisor_reply(
                "Welcome the customer and ask their name in one short message.",
                "hii",
            )
        self.assertNotIn("welcome the customer", reply.casefold())
        self.assertNotIn("ask their name", reply.casefold())
        self.assertTrue(agent._safe_customer_reply(reply))

    async def test_advisor_reply_uses_customer_ready_fallback_when_rate_limited(self):
        fallback = "Hi! Welcome to Pratap AI Property Advisor. What name should I use for you?"
        with patch.object(
            agent, "_create_completion", AsyncMock(side_effect=RuntimeError("rate limited"))
        ):
            reply = await agent.generate_advisor_reply(fallback, "hii")
        self.assertEqual(reply, fallback)

    async def test_rate_limited_identity_does_not_treat_greeting_as_name(self):
        with patch.object(
            agent, "_create_completion", AsyncMock(side_effect=RuntimeError("rate limited"))
        ):
            identity = await agent.extract_customer_identity("hii")
        self.assertIsNone(identity["name"])

    async def test_finalizer_reuses_existing_llm_answer_without_second_call(self):
        writer = AsyncMock()
        with patch.object(agent, "generate_advisor_reply", writer):
            reply = await agent.finalize_agent_answer(
                {"answer": "Sure—what would you like to know?", "answer_ready": True},
                "Tell me more",
            )
        writer.assert_not_awaited()
        self.assertEqual(reply, "Sure—what would you like to know?")

    async def test_finalizer_writes_database_result_once(self):
        writer = AsyncMock(return_value="I found five Pune homes. Want to see them?")
        with patch.object(agent, "generate_advisor_reply", writer):
            reply = await agent.finalize_agent_answer(
                {"answer": "There are 5 matching properties in Pune."},
                "How many properties are in Pune?",
            )
        writer.assert_awaited_once()
        self.assertIn("five Pune homes", reply)

    async def test_property_caption_is_llm_written_with_locked_facts(self):
        generated = (
            "🏡 Aster Koramangala 173\n📍 Koramangala, Bengaluru\n"
            "3 BHK · 1,850 sq ft\n💰 ₹1.8 crore\nLooks promising? I can share details."
        )
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=__import__("json").dumps([generated]))
        )])
        card = {
            "title": "Aster Koramangala 173",
            "location": "Koramangala, Bengaluru",
            "configuration": "3 BHK", "area": "1,850 sq ft",
            "price": "₹1.8 crore", "fallback": "fallback caption",
        }
        with patch.object(agent, "_create_completion", AsyncMock(return_value=completion)):
            captions = await agent.generate_property_captions([card], "This looks nice")
        self.assertEqual(captions, [generated])

    async def test_property_caption_falls_back_if_llm_changes_a_fact(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='["Aster Koramangala 173 for ₹2 crore"]')
        )])
        card = {
            "title": "Aster Koramangala 173",
            "location": "Koramangala, Bengaluru",
            "configuration": "3 BHK", "area": "1,850 sq ft",
            "price": "₹1.8 crore", "fallback": "verified fallback caption",
        }
        with patch.object(agent, "_create_completion", AsyncMock(return_value=completion)):
            captions = await agent.generate_property_captions([card], "Show me this")
        self.assertEqual(captions, ["verified fallback caption"])

    async def test_property_facts_are_answered_from_db_when_llm_is_unavailable(self):
        state = {
            "message": "Does it have parking and what is maintenance?",
            "intent": {"intent": "property_question"},
            "matches": [{
                "title": "Aster Koramangala 173", "parking_spaces": 1,
                "maintenance_monthly": 3900, "currency": "INR",
            }],
            "page_info": {"focused_property": True}, "knowledge": [], "history": [],
        }
        with patch.object(
            agent.llm.chat.completions, "create",
            AsyncMock(side_effect=RuntimeError("429 free quota exhausted")),
        ):
            result = await agent.compose_answer(state)
        self.assertIn("Parking spaces: 1", result["answer"])
        self.assertIn("₹3,900", result["answer"])
        self.assertEqual(result["model"], "database-fallback")

    async def test_booking_and_cancellation_work_without_llm(self):
        with patch.object(
            agent, "retrieve_knowledge", AsyncMock(return_value={"knowledge": []})
        ), patch.object(
            agent.llm.chat.completions, "create",
            AsyncMock(side_effect=RuntimeError("429 free quota exhausted")),
        ):
            booking = await agent.advise_on_selected_property(
                "Please schedule a call", {"title": "Aster Koramangala 173"}
            )
            cancelled = await agent.classify_booking_cancellation(
                "Please cancel my consultation", {"status": "confirmed"}
            )
        self.assertEqual(booking["action"], "book")
        self.assertTrue(cancelled)

    async def test_clear_booking_questions_and_deferrals_are_deterministic(self):
        listing = {
            "title": "Aravalli Heights", "parking_spaces": 2,
            "images": ["one", "two"],
        }
        completion = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        knowledge = AsyncMock(side_effect=AssertionError("KB should not override property facts"))
        with patch.object(agent.llm.chat.completions, "create", completion), patch.object(
            agent, "retrieve_knowledge", knowledge
        ):
            consultation = await agent.advise_on_selected_property(
                "Can we schedule a consultation?", listing
            )
            visit = await agent.advise_on_selected_property(
                "I want to visit the property this Saturday", listing
            )
            parking = await agent.advise_on_selected_property(
                "Does it have parking?", listing
            )
            pictures = await agent.advise_on_selected_property(
                "Show me more pictures", listing
            )
            later = await agent.advise_on_selected_property(
                "I am not ready to book", listing
            )
        self.assertEqual(consultation["action"], "book")
        self.assertEqual(visit["action"], "book")
        self.assertIn("Parking spaces: 2", parking["answer"])
        self.assertTrue(pictures["wants_images"])
        self.assertEqual(later["action"], "decline")
        completion.assert_not_called()
        knowledge.assert_not_called()

    async def test_natural_property_selection_works_without_llm(self):
        with patch.object(
            agent.llm.chat.completions, "create",
            AsyncMock(side_effect=RuntimeError("provider unavailable")),
        ):
            named = await agent.classify_property_response(
                "I like Ivy Sarjapur Road 104",
                ["Aster Koramangala 173", "Ivy Sarjapur Road 104"],
            )
            ordinal = await agent.classify_property_response(
                "The second one looks perfect",
                ["Aster Koramangala 173", "Ivy Sarjapur Road 104"],
            )
            ambiguous = await agent.classify_property_response(
                "That one looks good",
                ["Aster Koramangala 173", "Ivy Sarjapur Road 104"],
            )
        self.assertEqual(named["selected_property"], "Ivy Sarjapur Road 104")
        self.assertEqual(ordinal["selected_property"], "Ivy Sarjapur Road 104")
        self.assertTrue(ambiguous["is_interested"])
        self.assertIsNone(ambiguous["selected_property"])

    async def test_single_city_returns_five_then_next_five_without_repeating(self):
        sessions = FakeSearchSessions()
        fake_db = SimpleNamespace(search_sessions=sessions)
        with patch.object(agent, "properties", FakeProperties()), patch.object(agent, "db", fake_db):
            first = await agent.search_inventory({
                "session_id": "one",
                "intent": {"intent": "property_search", "city": "Bengaluru"},
            })
            second = await agent.search_inventory({
                "session_id": "one",
                "intent": {"intent": "property_search", "_next_page": True, "_current_filters": {}},
            })
        self.assertEqual(len(first["matches"]), 5)
        self.assertEqual(len(second["matches"]), 3)
        self.assertTrue(
            set(row["title"] for row in first["matches"]).isdisjoint(
                row["title"] for row in second["matches"]
            )
        )

    async def test_two_cities_return_three_from_each(self):
        sessions = FakeSearchSessions()
        with patch.object(agent, "properties", FakeProperties()), patch.object(
            agent, "db", SimpleNamespace(search_sessions=sessions)
        ):
            result = await agent.search_inventory({
                "session_id": "two",
                "intent": {"intent": "property_search", "cities": ["Pune", "Mumbai"]},
            })
        self.assertEqual(len(result["matches"]), 6)
        self.assertEqual(result["page_info"]["per_city_shown"], {"Pune": 3, "Mumbai": 3})

    async def test_broad_buy_request_asks_for_city_instead_of_dumping_inventory(self):
        sessions = FakeSearchSessions()
        with patch.object(agent, "properties", FakeProperties()), patch.object(
            agent, "db", SimpleNamespace(search_sessions=sessions)
        ):
            result = await agent.search_inventory({
                "session_id": "broad",
                "intent": {"intent": "property_search", "purpose": "buy"},
            })
        self.assertEqual(result["matches"], [])
        self.assertTrue(result["page_info"]["requires_city"])

    async def test_saved_lead_restores_property_context_for_later_questions(self):
        knowledge = SimpleNamespace(find=lambda *_args, **_kwargs: FakeCursor([]))
        pending = SimpleNamespace(find_one=AsyncMock(return_value=None))
        leads = SimpleNamespace(find_one=AsyncMock(return_value={
            "property_title": "Aster Koramangala 173"
        }))
        property_store = SimpleNamespace(find_one=AsyncMock(return_value={
            "title": "Aster Koramangala 173", "city": "Bengaluru",
            "images": ["one", "two"],
        }))
        with patch.object(
            agent, "db", SimpleNamespace(
                knowledge_base=knowledge, pending_leads=pending, leads=leads
            )
        ), patch.object(agent, "properties", property_store):
            result = await agent.retrieve_knowledge({
                "session_id": "wa:918815096521",
                "message": "Does it have more photos?",
                "history": [], "intent": {"intent": "general"}, "matches": [],
            })
        self.assertEqual(result["matches"][0]["title"], "Aster Koramangala 173")
        self.assertTrue(result["page_info"]["focused_property"])


if __name__ == "__main__":
    unittest.main()
