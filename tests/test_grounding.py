import unittest
from unittest.mock import AsyncMock, patch

from app import agent
from app.grounding import cosine_similarity, evidence_answer


class GroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_inventory_names_city_and_only_offers_live_nearby_cities(self):
        state = {
            "message": "Show me properties in Delhi",
            "intent": {"intent": "property_search"},
            "matches": [],
            "knowledge": [],
            "history": [],
            "page_info": {"exhausted": True, "cities": ["Delhi"], "offset": 0},
        }
        with patch.object(
            agent,
            "_available_inventory_cities",
            AsyncMock(return_value=["Bengaluru", "Gurugram", "Noida"]),
        ):
            result = await agent.compose_answer(state)

        self.assertEqual(result["model"], "database")
        self.assertIn("don't have any properties available in Delhi", result["answer"])
        self.assertIn("Noida and Gurugram", result["answer"])

    async def test_nearby_fallback_explains_listings_are_outside_requested_city(self):
        result = await agent.compose_answer({
            "message": "Show me properties in Delhi",
            "intent": {"intent": "property_search"},
            "matches": [
                {"title": "Noida Home", "city": "Noida"},
                {"title": "Gurugram Home", "city": "Gurugram"},
            ],
            "knowledge": [],
            "history": [],
            "page_info": {
                "total": 12,
                "cities": ["Noida", "Gurugram"],
                "nearby_fallback": True,
                "requested_city": "Delhi",
            },
        })

        self.assertEqual(result["model"], "database")
        self.assertIn("don't have any properties available in Delhi", result["answer"])
        self.assertIn("nearby options from Noida and Gurugram", result["answer"])

    def test_cosine_similarity_ranks_identical_evidence(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_evidence_answer_uses_saved_text_and_source(self):
        answer = evidence_answer([{
            "content": "Only this verified policy is allowed.",
            "source": "CRM policy",
        }])
        self.assertEqual(answer, "Only this verified policy is allowed. Source: CRM policy.")

    async def test_strict_mode_refuses_when_there_is_no_evidence(self):
        result = await agent.compose_answer({
            "message": "Will this unknown project double in value next year?",
            "intent": {"intent": "general"},
            "matches": [], "knowledge": [], "history": [], "page_info": {},
        })
        self.assertEqual(result["model"], "grounding-guard")
        self.assertIn("don't have verified information", result["answer"])
        self.assertFalse(result["grounded"])

    async def test_knowledge_answer_is_extractive_not_model_generated(self):
        result = await agent.compose_answer({
            "message": "What is your listing policy?",
            "intent": {"intent": "general"},
            "matches": [],
            "knowledge": [{
                "title": "Listing policy", "content": "Prices come from the live database.",
                "source": "EstateAI policy", "retrieval_method": "embedding",
                "retrieval_score": 0.91,
            }],
            "history": [], "page_info": {},
        })
        self.assertEqual(result["model"], "knowledge-base")
        self.assertEqual(
            result["answer"],
            "Prices come from the live database. Source: EstateAI policy.",
        )
        self.assertTrue(result["grounded"])


if __name__ == "__main__":
    unittest.main()
