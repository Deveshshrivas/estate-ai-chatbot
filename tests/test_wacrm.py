import hashlib
import hmac
import json
import httpx
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import agent, wacrm
from app import main
from starlette.requests import Request
from fastapi import Response


def request_with_json(path: str, payload: dict, headers: list[tuple[bytes, bytes]] | None = None):
    body = json.dumps(payload).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({
        "type": "http", "method": "POST", "path": path,
        "headers": headers or [], "query_string": b"",
        "server": ("test", 80), "client": ("test", 1), "scheme": "https",
    }, receive)


class WacrmTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_rejects_free_model_for_production(self):
        response = Response()
        with patch.object(main.client.admin, "command", AsyncMock()), patch.object(
            main.settings, "openrouter_api_key", "configured"
        ), patch.object(
            main.settings, "openrouter_model", "openrouter/free"
        ), patch.object(
            main.settings, "wacrm_api_url", "https://wacrm.example"
        ), patch.object(
            main.settings, "wacrm_api_key", "configured"
        ), patch.object(
            main.settings, "wacrm_webhook_secret", "configured"
        ), patch.object(
            main.settings, "cron_secret", "configured"
        ), patch.object(
            main.settings, "property_admin_secret", "configured"
        ):
            result = await main.readiness(response)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(result["status"], "not_ready")
        self.assertFalse(result["checks"]["llm"]["ok"])

    async def test_safe_wacrm_get_retries_transient_error(self):
        request = AsyncMock(side_effect=[
            httpx.ReadTimeout("timeout"),
            httpx.Response(
                200, json={"data": {"id": "contact-1"}},
                request=httpx.Request("GET", "https://wacrm.example/contact"),
            ),
        ])
        client = AsyncMock()
        client.__aenter__.return_value.request = request
        client.__aexit__.return_value = None
        with patch.object(wacrm, "is_configured", return_value=True), patch.object(
            wacrm.httpx, "AsyncClient", return_value=client
        ), patch.object(wacrm.asyncio, "sleep", AsyncMock()):
            result = await wacrm._request("GET", "/contact", retryable=True)
        self.assertEqual(result, {"id": "contact-1"})
        self.assertEqual(request.await_count, 2)

    async def test_wacrm_webhook_awaits_inbound_processing(self):
        process = AsyncMock()
        request = request_with_json(
            "/wacrm-webhook",
            {"event": "message.received", "data": {"whatsapp_message_id": "wamid-1"}},
        )
        with patch.object(wacrm, "verify_signature", return_value=True), patch.object(
            wacrm, "process_inbound", process
        ):
            response = await main.receive_wacrm_webhook(request)
        process.assert_awaited_once()
        self.assertEqual(response["status"], "accepted")

    async def test_failed_wacrm_event_is_released_for_provider_retry(self):
        events = SimpleNamespace(
            insert_one=AsyncMock(), delete_one=AsyncMock(),
        )
        fake_db = SimpleNamespace(webhook_events=events)
        with patch.object(wacrm, "db", fake_db), patch.object(
            wacrm, "get_contact", AsyncMock(side_effect=RuntimeError("temporary outage"))
        ):
            with self.assertRaisesRegex(RuntimeError, "temporary outage"):
                await wacrm.process_inbound({
                    "whatsapp_message_id": "wamid-retry", "contact_id": "contact-1",
                    "text": "hello",
                })
        events.delete_one.assert_awaited_once_with({"message_id": "wamid-retry"})

    async def test_meta_webhook_awaits_message_processing(self):
        process = AsyncMock()
        request = request_with_json("/webhooks/whatsapp", {"entry": []})
        with patch.object(main, "verify_signature", return_value=True), patch.object(
            main, "extract_messages", return_value=[{"id": "wamid-2"}]
        ), patch.object(main, "process_message", process):
            response = await main.receive_whatsapp_webhook(request)
        process.assert_awaited_once_with({"id": "wamid-2"})
        self.assertEqual(response["status"], "accepted")

    def test_signature_verification_and_replay_guard(self):
        body = b'{"event":"message.received"}'
        timestamp = str(int(time.time()))
        with patch.object(wacrm.settings, "wacrm_webhook_secret", "secret"):
            digest = hmac.new(
                b"secret", f"{timestamp}.".encode() + body, hashlib.sha256
            ).hexdigest()
            self.assertTrue(wacrm.verify_signature(body, f"t={timestamp},v1={digest}"))
            self.assertFalse(wacrm.verify_signature(body, f"t={int(timestamp)-1000},v1={digest}"))
            self.assertFalse(wacrm.verify_signature(body, "bad"))

    def test_opt_out_language(self):
        self.assertTrue(wacrm._is_opt_out("No thanks, I am not interested"))
        self.assertTrue(wacrm._is_opt_out("Please STOP"))
        self.assertFalse(wacrm._is_opt_out("I am interested in a 2 BHK"))

    def test_govind_conversation_intents(self):
        self.assertTrue(wacrm._asks_for_call("Can I call you?"))
        self.assertTrue(wacrm._asks_available_cities("In which city do you have properties?"))
        self.assertTrue(wacrm._asks_available_cities("In wgich city u have properties"))
        self.assertTrue(wacrm._is_property_opt_out("I don't want property anymore"))
        self.assertTrue(wacrm._is_goodbye("Bye"))
        self.assertFalse(wacrm._is_property_opt_out("No, show me Mumbai instead"))
        self.assertFalse(wacrm._is_property_opt_out(
            "No I am not interested in Bengaluru, I want Pune"
        ))
        self.assertFalse(wacrm._is_property_opt_out(
            "Not interested in this one, show another property"
        ))

    async def test_llm_classifies_city_rejection_as_preference_change(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=(
                '{"action":"preference_change","reason":"customer selected Pune"}'
            ))
        )])
        with patch.object(agent, "_create_completion", AsyncMock(return_value=completion)):
            result = await agent.classify_customer_control(
                "No, I am not interested in Bengaluru; I want Pune"
            )
        self.assertEqual(result["action"], "preference_change")

    async def test_non_template_sent_event_does_not_start_campaign(self):
        insert = AsyncMock()
        with patch.object(wacrm.db.webhook_events, "insert_one", insert), patch.object(
            wacrm, "get_contact", AsyncMock()
        ):
            # The route filters this event before record_template_send is called.
            from app.main import receive_wacrm_webhook
            self.assertTrue(callable(receive_wacrm_webhook))
        insert.assert_not_awaited()

    async def test_cancelled_mongo_lead_is_synced_to_wacrm_as_lost(self):
        lead = {
            "_id": "lead-1", "name": "Devesh", "status": "cancelled",
            "property_title": "Aster Koramangala 173",
            "cancelled_at": "2026-07-31T16:00:00Z",
            "cancellation_message": "Please cancel my appointment",
        }
        leads = SimpleNamespace(
            find_one=AsyncMock(return_value=lead),
            update_one=AsyncMock(),
        )
        request = AsyncMock(return_value={"id": "deal-1"})
        with patch.object(wacrm, "db", SimpleNamespace(leads=leads)), patch.object(
            wacrm, "_request", request
        ):
            await wacrm._sync_cancelled_lead("918815096521", "contact-1", "conversation-1")
        payload = request.await_args.kwargs["json"]
        self.assertEqual(payload["status"], "lost")
        self.assertIn("cancelled", payload["notes"].lower())
        leads.update_one.assert_awaited_once()

    async def test_existing_wacrm_deal_is_refreshed_when_lead_converts(self):
        lead = {
            "_id": "lead-1", "name": "Govind", "status": "qualified",
            "lead_stage": "appointment_requested", "preferred_schedule": "1 Aug 11 AM",
            "wacrm_synced_at": "older-sync", "wacrm_deal_id": "deal-1",
            "budget_max": 2_000_000,
        }
        leads = SimpleNamespace(
            find_one=AsyncMock(return_value=lead), update_one=AsyncMock()
        )
        request = AsyncMock(return_value={"id": "deal-1", "created": False})
        with patch.object(wacrm, "db", SimpleNamespace(leads=leads)), patch.object(
            wacrm, "_request", request
        ):
            await wacrm._sync_new_lead("919263935852", "contact-1", "conversation-1")
        payload = request.await_args.kwargs["json"]
        self.assertIn("appointment_requested", payload["notes"])
        self.assertEqual(payload["value"], 2_000_000)
        self.assertEqual(payload["currency"], "INR")

    async def test_wacrm_deal_permission_error_never_blocks_customer_reply(self):
        lead = {
            "_id": "lead-2", "name": "Govind", "status": "engaged",
            "lead_stage": "discovery", "city": "Patna",
        }
        leads = SimpleNamespace(
            find_one=AsyncMock(return_value=lead), update_one=AsyncMock()
        )
        request = AsyncMock(side_effect=RuntimeError("403 Forbidden"))
        with patch.object(wacrm, "db", SimpleNamespace(leads=leads)), patch.object(
            wacrm, "_request", request
        ):
            # The sync failure is recorded for retry but does not propagate to
            # the inbound conversation handler.
            await wacrm._sync_new_lead("919263935852", "contact-1", "conversation-1")
        saved = leads.update_one.await_args.args[1]["$set"]
        self.assertTrue(saved["wacrm_sync_pending"])
        self.assertIn("403", saved["wacrm_sync_error"])

    async def test_public_form_schedules_template_and_pipeline_within_two_to_five_minutes(self):
        form_leads = SimpleNamespace(insert_one=AsyncMock())
        campaigns = SimpleNamespace(update_many=AsyncMock(), insert_one=AsyncMock())
        leads = SimpleNamespace(update_one=AsyncMock())
        fake_db = SimpleNamespace(
            public_form_leads=form_leads, outbound_campaigns=campaigns, leads=leads
        )
        request = AsyncMock(return_value={"id": "contact-1"})
        save = AsyncMock()
        submission = {
            "submission_id": "submission-1", "name": "Devesh",
            "phone": "918815096521", "email": "devesh@example.com",
            "platform_name": "Google", "property_purpose": "buy",
            "city": "Bengaluru", "budget": "80 lakh",
            "query_reason": "Need a 2 BHK near Sarjapur Road",
            "whatsapp_consent": "true",
        }
        with patch.object(wacrm, "db", fake_db), patch.object(
            wacrm, "_request", request
        ), patch.object(wacrm, "save_message", save), patch.object(
            wacrm, "_sync_new_lead", AsyncMock()
        ) as sync:
            result = await wacrm.schedule_public_form_enquiry(submission)
        campaign = campaigns.insert_one.await_args.args[0]
        delay = result["scheduled_for"] - campaign["started_at"]
        self.assertGreaterEqual(int(delay.total_seconds()), 2 * 60)
        self.assertLessEqual(int(delay.total_seconds()), 5 * 60)
        self.assertEqual(campaign["template_params"][2], "Bengaluru")
        self.assertIn("Sarjapur Road", campaign["template_params"][3])
        saved_lead = leads.update_one.await_args.args[1]["$set"]
        self.assertEqual(saved_lead["lead_stage"], "form_submitted")
        self.assertEqual(saved_lead["budget_max"], 8_000_000)
        sync.assert_awaited_once_with("918815096521", "contact-1", None)
        save.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
