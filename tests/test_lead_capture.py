import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import whatsapp


class EarlyLeadCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_booking_stops_campaign_and_form_followups(self):
        campaigns = SimpleNamespace(update_many=AsyncMock())
        forms = SimpleNamespace(update_many=AsyncMock())
        with patch.object(
            whatsapp, "db",
            SimpleNamespace(outbound_campaigns=campaigns, public_form_leads=forms),
        ):
            await whatsapp.stop_sales_followups("918815096521")
        campaign_update = campaigns.update_many.await_args.args[1]["$set"]
        self.assertEqual(campaign_update["status"], "converted")
        self.assertIsNone(campaign_update["next_followup_at"])
        forms.update_many.assert_awaited_once()

    async def test_complete_callback_details_then_collect_schedule(self):
        pending = {
            "_id": "pending-1", "session_id": "wa:919999999999",
            "stage": "awaiting_details", "selected_property": "Aravalli Heights",
            "matched_properties": ["Aravalli Heights"],
        }
        pending_leads = SimpleNamespace(
            find_one=AsyncMock(return_value=pending), update_one=AsyncMock()
        )
        send = AsyncMock()
        database = SimpleNamespace(pending_leads=pending_leads)
        with patch.object(whatsapp, "db", database):
            handled = await whatsapp._handle_lead_details(
                "919999999999", "Govind",
                "Govind | govind@example.com | 919999999999 | PHONE CALL",
                send=send,
            )
        self.assertTrue(handled)
        saved = pending_leads.update_one.await_args.args[1]["$set"]
        self.assertEqual(saved["stage"], "awaiting_call_schedule")
        self.assertEqual(saved["email"], "govind@example.com")
        self.assertIn("date and time", send.await_args.args[1].lower())

    async def test_qualified_appointment_request_can_be_cancelled(self):
        booking = {
            "_id": "lead-1", "whatsapp_phone": "919999999999",
            "status": "qualified", "appointment_status": "requested",
            "property_title": "Aravalli Heights",
        }
        leads = SimpleNamespace(
            find_one=AsyncMock(return_value=booking), update_one=AsyncMock()
        )
        campaigns = SimpleNamespace(update_many=AsyncMock())
        visits = SimpleNamespace(update_many=AsyncMock())
        send = AsyncMock()
        database = SimpleNamespace(
            leads=leads, outbound_campaigns=campaigns, site_visits=visits
        )
        with patch.object(whatsapp, "db", database), patch.object(
            whatsapp, "classify_booking_cancellation", AsyncMock(return_value=True)
        ):
            handled = await whatsapp._cancel_saved_booking(
                "919999999999", "Please cancel my appointment", send=send
            )
        self.assertTrue(handled)
        query = leads.find_one.await_args.args[0]
        self.assertIn("appointment_status", query["$or"][0])
        saved = leads.update_one.await_args.args[1]["$set"]
        self.assertEqual(saved["status"], "cancelled")
        self.assertEqual(saved["appointment_status"], "cancelled")
        send.assert_awaited_once()

    async def test_callback_date_without_time_is_saved_then_time_is_requested(self):
        pending = {
            "_id": "pending-1", "session_id": "wa:919999999999",
            "stage": "awaiting_call_schedule", "whatsapp_name": "Govind",
        }
        pending_leads = SimpleNamespace(
            find_one=AsyncMock(return_value=pending), update_one=AsyncMock()
        )
        send = AsyncMock()
        with patch.object(whatsapp, "db", SimpleNamespace(pending_leads=pending_leads)):
            handled = await whatsapp._handle_lead_details(
                "919999999999", "Govind", "1 Aug", send=send
            )
        self.assertTrue(handled)
        update = pending_leads.update_one.await_args.args[1]["$set"]
        self.assertEqual(update["preferred_date"], "1 Aug")
        self.assertIn("time", send.await_args.args[1].lower())

    async def test_property_search_creates_lead_before_appointment(self):
        leads = SimpleNamespace(
            find_one=AsyncMock(return_value=None),
            update_one=AsyncMock(),
        )
        with patch.object(whatsapp, "db", SimpleNamespace(leads=leads)):
            captured = await whatsapp.capture_engaged_lead(
                phone="918815096521",
                name="Devesh",
                intent={
                    "intent": "property_search",
                    "purpose": "buy",
                    "city": "Bengaluru",
                    "max_price": 8_000_000,
                },
                matches=[{"title": "Ivy Sarjapur Road 104"}],
                last_message="I need a flat in Bengaluru",
            )
        self.assertTrue(captured)
        update = leads.update_one.await_args.args[1]["$set"]
        self.assertEqual(update["status"], "engaged")
        self.assertEqual(update["appointment_status"], "not_booked")
        self.assertEqual(update["budget_currency"], "INR")
        self.assertEqual(update["matched_properties"], ["Ivy Sarjapur Road 104"])

    async def test_greeting_alone_does_not_create_lead(self):
        leads = SimpleNamespace(
            find_one=AsyncMock(),
            update_one=AsyncMock(),
        )
        with patch.object(whatsapp, "db", SimpleNamespace(leads=leads)):
            captured = await whatsapp.capture_engaged_lead(
                phone="918815096521",
                intent={"intent": "greeting"},
                last_message="Hi",
            )
        self.assertFalse(captured)
        leads.update_one.assert_not_awaited()

    async def test_property_selection_qualifies_same_lead_without_booking(self):
        pending = {
            "_id": "pending-1",
            "session_id": "wa:918815096521",
            "stage": "showing_results",
            "matched_properties": ["Ivy Sarjapur Road 104"],
            "preferences": {"city": "Bengaluru", "purpose": "buy"},
        }
        pending_leads = SimpleNamespace(
            find_one=AsyncMock(return_value=pending),
            update_one=AsyncMock(),
        )
        leads = SimpleNamespace(update_one=AsyncMock())
        send = AsyncMock()
        decision = {"is_interested": True, "selected_property": "Ivy Sarjapur Road 104"}
        with patch.object(
            whatsapp,
            "db",
            SimpleNamespace(pending_leads=pending_leads, leads=leads),
        ), patch.object(
            whatsapp, "classify_property_response", AsyncMock(return_value=decision)
        ):
            handled = await whatsapp._handle_lead_details(
                "918815096521", "Devesh", "This Ivy property looks good", send=send
            )
        self.assertTrue(handled)
        lead_update = leads.update_one.await_args.args[1]["$set"]
        self.assertEqual(lead_update["status"], "interested")
        self.assertEqual(lead_update["appointment_status"], "not_booked")
        self.assertEqual(lead_update["property_title"], "Ivy Sarjapur Road 104")

    async def test_first_property_question_fetches_database_row_and_answers_it(self):
        pending = {
            "_id": "pending-1", "session_id": "wa:918815096521",
            "stage": "showing_results",
            "matched_properties": ["Ivy Sarjapur Road 104"],
            "preferences": {"city": "Bengaluru"},
        }
        pending_leads = SimpleNamespace(
            find_one=AsyncMock(return_value=pending), update_one=AsyncMock()
        )
        leads = SimpleNamespace(update_one=AsyncMock())
        properties = SimpleNamespace(find_one=AsyncMock(return_value={
            "title": "Ivy Sarjapur Road 104", "parking_spaces": 2,
            "maintenance_monthly": 5300, "images": [],
        }))
        send = AsyncMock()
        decision = {
            "is_interested": True, "selected_property": "Ivy Sarjapur Road 104",
            "is_question": True,
        }
        advice = {"action": "question", "wants_images": False, "answer": "It has 2 parking spaces."}
        with patch.object(
            whatsapp, "db", SimpleNamespace(
                pending_leads=pending_leads, leads=leads, properties=properties
            )
        ), patch.object(
            whatsapp, "classify_property_response", AsyncMock(return_value=decision)
        ), patch.object(
            whatsapp, "advise_on_selected_property", AsyncMock(return_value=advice)
        ):
            handled = await whatsapp._handle_lead_details(
                "918815096521", "Devesh",
                "Does Ivy Sarjapur Road 104 have parking?", send=send,
            )
        self.assertTrue(handled)
        properties.find_one.assert_awaited_once()
        self.assertIn("2 parking spaces", send.await_args.args[1])


if __name__ == "__main__":
    unittest.main()
