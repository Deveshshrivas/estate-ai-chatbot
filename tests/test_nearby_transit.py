import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app import agent, nearby_transit


class NearbyTransitTests(unittest.IsolatedAsyncioTestCase):
    def test_distance_and_nearest_station_are_calculated_from_coordinates(self):
        elements = [
            {"type": "node", "id": 1, "lat": 12.99, "lon": 77.63,
             "tags": {"name": "Far Metro"}},
            {"type": "node", "id": 2, "lat": 12.94, "lon": 77.62,
             "tags": {"name": "Near Metro", "network": "Namma Metro"}},
        ]
        result = nearby_transit.nearest_from_elements(elements, 12.9354, 77.6247)
        self.assertEqual(result["name"], "Near Metro")
        self.assertLess(result["distance_km"], 1)
        self.assertEqual(result["source"], "OpenStreetMap")

    async def test_fresh_cached_station_does_not_call_external_service(self):
        cached = {
            "status": "found", "name": "Cached Metro", "distance_km": 1.2,
            "fetched_at": datetime.now(timezone.utc),
        }
        with patch.object(
            nearby_transit, "fetch_nearest_metro", AsyncMock()
        ) as fetch:
            result = await nearby_transit.nearest_metro_for_property({
                "title": "Test Home", "nearby_metro": cached,
                "coordinates": {"latitude": 12.9, "longitude": 77.6},
            })
        self.assertEqual(result["name"], "Cached Metro")
        fetch.assert_not_awaited()

    async def test_agent_enriches_metro_question_and_uses_grounded_answer(self):
        metro = {
            "status": "found", "name": "Indiranagar", "distance_km": 1.35,
            "network": "Namma Metro", "source": "OpenStreetMap",
        }
        state = {
            "message": "Which metro is nearest?",
            "intent": {"intent": "property_question"},
            "matches": [{"title": "Test Home"}],
            "page_info": {"focused_property": True},
        }
        with patch.object(
            agent, "nearest_metro_for_property", AsyncMock(return_value=metro)
        ):
            enriched = await agent.enrich_nearby_transit(state)
        answer = agent._database_property_answer(
            state["message"], enriched["matches"][0]
        )
        self.assertIn("Indiranagar", answer)
        self.assertIn("1.35 km", answer)
        self.assertIn("OpenStreetMap", answer)

    async def test_missing_coordinates_returns_honest_answer(self):
        result = await nearby_transit.nearest_metro_for_property({"title": "No Map Home"})
        answer = agent._database_property_answer(
            "Is a metro nearby?", {"title": "No Map Home", "nearby_metro": result}
        )
        self.assertEqual(result["status"], "coordinates_missing")
        self.assertIn("no coordinates", answer)


if __name__ == "__main__":
    unittest.main()
