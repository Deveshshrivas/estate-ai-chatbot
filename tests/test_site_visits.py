import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from app import site_visits
from app.site_visits import INDIA, parse_requested_slot


class SiteVisitSlotTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 7, 12, 0, tzinfo=INDIA)

    def test_understands_tomorrow_with_time(self):
        slot, has_date, has_time = parse_requested_slot(
            "Can I visit tomorrow at 11 am?", now=self.now
        )
        self.assertTrue(has_date)
        self.assertTrue(has_time)
        self.assertEqual(slot.astimezone(INDIA).strftime("%Y-%m-%d %H:%M"), "2026-08-08 11:00")

    def test_asks_for_time_when_only_date_is_given(self):
        slot, has_date, has_time = parse_requested_slot("10 August", now=self.now)
        self.assertIsNone(slot)
        self.assertTrue(has_date)
        self.assertFalse(has_time)

    def test_combines_saved_date_and_next_time_reply(self):
        slot, has_date, has_time = parse_requested_slot(
            "3:30 pm", saved_date="10 August", now=self.now
        )
        self.assertTrue(has_date and has_time)
        self.assertEqual(slot.astimezone(INDIA).strftime("%Y-%m-%d %H:%M"), "2026-08-10 15:30")

    def test_understands_misspelled_day_after_tomorrow(self):
        slot, has_date, has_time = parse_requested_slot(
            "day after tommorow at 4 pm", now=self.now
        )
        self.assertTrue(has_date and has_time)
        self.assertEqual(slot.astimezone(INDIA).strftime("%Y-%m-%d %H:%M"), "2026-08-09 16:00")

    def test_understands_ordinal_this_month(self):
        slot, has_date, has_time = parse_requested_slot(
            "31st this month at 11 am", now=self.now
        )
        self.assertTrue(has_date and has_time)
        self.assertEqual(slot.astimezone(INDIA).strftime("%Y-%m-%d %H:%M"), "2026-08-31 11:00")


class SiteVisitAssignmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_visit_is_automatically_assigned_to_available_salesperson(self):
        collection = SimpleNamespace(
            insert_one=AsyncMock(return_value=SimpleNamespace(inserted_id="visit-1"))
        )
        available = [{"id": "sales-1", "name": "Asha", "email": "asha@example.com"}]
        with patch.object(
            site_visits, "available_salespeople", AsyncMock(return_value=available)
        ), patch.object(
            site_visits, "db", SimpleNamespace(site_visits=collection)
        ):
            result = await site_visits.create_site_visit(
                phone="919999999999", customer_name="Demo Customer",
                customer_email="demo@example.com", property_title="Demo Property",
                scheduled_at=datetime(2026, 8, 15, 11, 0, tzinfo=INDIA),
            )
        self.assertEqual(result["owner_id"], "sales-1")
        self.assertEqual(result["owner"], "Asha")


if __name__ == "__main__":
    unittest.main()
