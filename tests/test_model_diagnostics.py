import json
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage

from harness_agent import core
from harness_agent import model_diagnostics


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, get_responses=None, post_responses=None):
        self.get_responses = list(get_responses or [])
        self.post_responses = list(post_responses or [])
        self.get_calls = []
        self.post_calls = []

    def get(self, url, headers=None, timeout=None):
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.get_responses.pop(0)

    def post(self, url, headers=None, json=None, timeout=None):
        self.post_calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.post_responses.pop(0)


class ModelDiagnosticsTests(unittest.TestCase):
    def test_model_status_reports_context_and_masks_api_key(self) -> None:
        fake = FakeSession(
            get_responses=[
                FakeResponse(
                    payload={"data": [{"id": "deepseek-v4-flash"}]},
                    headers={"x-ratelimit-remaining-tokens": "123", "x-ratelimit-reset-tokens": "30s"},
                ),
                FakeResponse(payload={"data": [{"id": "deepseek-v4-flash", "model_info": {"max_input_tokens": 128000}}]}),
            ]
        )
        history = [SystemMessage(content="system prompt"), HumanMessage(content="hello " * 100)]

        with patch.object(core, "LITELLM_BASE_URL", "http://proxy.test/v1"), patch.object(core, "LITELLM_API_KEY", "sk-secret-value"):
            payload = json.loads(model_diagnostics.model_status_report("deepseek-v4-flash", history, session=fake))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["proxy"]["rate_limits"]["models"]["remaining_tokens"], "123")
        self.assertEqual(payload["proxy"]["rate_limits"]["models"]["reset_tokens"], "30s")
        self.assertEqual(payload["api_key"]["masked"], "sk-s********lue")
        self.assertNotIn("sk-secret-value", json.dumps(payload))
        model = payload["models"][0]
        self.assertEqual(model["model"], "deepseek-v4-flash")
        self.assertEqual(model["context_window"], 128000)
        self.assertGreater(model["estimated_remaining_tokens"], 0)
        self.assertEqual(model["context_source"], "model_info")

    def test_smoke_test_model_reports_usage_and_rate_limit_headers(self) -> None:
        fake = FakeSession(
            post_responses=[
                FakeResponse(
                    payload={
                        "choices": [{"message": {"content": "hi"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                    headers={
                        "x-ratelimit-remaining-tokens": "999",
                        "x-ratelimit-reset-tokens": "12s",
                    },
                )
            ]
        )

        with patch.object(core, "LITELLM_BASE_URL", "http://proxy.test/v1"), patch.object(core, "LITELLM_API_KEY", "sk-secret-value"):
            result = model_diagnostics.smoke_test_model("deepseek-v4-flash", session=fake)

        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "deepseek-v4-flash")
        self.assertEqual(result["output"], "hi")
        self.assertEqual(result["usage"]["total_tokens"], 2)
        self.assertEqual(result["rate_limits"]["remaining_tokens"], "999")
        self.assertEqual(result["rate_limits"]["reset_tokens"], "12s")
        self.assertEqual(fake.post_calls[0]["json"]["messages"][0]["content"], "hi")
        self.assertEqual(fake.post_calls[0]["json"]["max_tokens"], 4)

    def test_model_test_report_can_test_all_listed_models(self) -> None:
        fake = FakeSession(
            get_responses=[FakeResponse(payload={"data": [{"id": "a"}, {"id": "b"}]})],
            post_responses=[
                FakeResponse(payload={"choices": [{"message": {"content": "ok-a"}}], "usage": {"total_tokens": 2}}),
                FakeResponse(status_code=429, payload={"error": {"message": "rate limited"}}, text="rate limited"),
            ],
        )

        with patch.object(core, "LITELLM_BASE_URL", "http://proxy.test/v1"), patch.object(core, "LITELLM_API_KEY", "sk-secret-value"):
            payload = json.loads(model_diagnostics.model_test_report("ignored", all_models=True, session=fake))

        self.assertFalse(payload["ok"])
        self.assertEqual([item["model"] for item in payload["results"]], ["a", "b"])
        self.assertTrue(payload["results"][0]["ok"])
        self.assertFalse(payload["results"][1]["ok"])
        self.assertEqual(payload["results"][1]["status_code"], 429)

    def test_missing_api_key_is_reported_without_calling_model(self) -> None:
        fake = FakeSession()

        with patch.object(core, "LITELLM_API_KEY", ""):
            result = model_diagnostics.smoke_test_model("deepseek-v4-flash", session=fake)

        self.assertFalse(result["ok"])
        self.assertIn("LITELLM_API_KEY", result["error"])
        self.assertEqual(fake.post_calls, [])

    def test_format_model_status_table_renders_human_readable_rows(self) -> None:
        payload = json.dumps({
            "ok": True,
            "proxy": {
                "base_url": "http://proxy.test/v1",
                "models_ok": True,
                "model_info_ok": True,
                "rate_limits": {
                    "models": {"remaining_tokens": "123", "reset_tokens": "30s"},
                    "model_info": {},
                },
            },
            "api_key": {"present": True, "masked": "sk-s********lue"},
            "active_model": "deepseek-v4-flash",
            "models": [
                {
                    "model": "deepseek-v4-flash",
                    "context_window": 128000,
                    "estimated_used_tokens": 9123,
                    "estimated_remaining_tokens": 118877,
                    "context_source": "model_info",
                    "estimate_method": "chars_div_4_plus_message_overhead",
                },
                {
                    "model": "groq-qwen3-32b",
                    "context_window": None,
                    "estimated_used_tokens": 9123,
                    "estimated_remaining_tokens": None,
                    "context_source": "unknown",
                    "estimate_method": "chars_div_4_plus_message_overhead",
                },
            ],
            "errors": {"models": "", "model_info": ""},
        })

        table = model_diagnostics.format_model_status_table(payload)

        self.assertIn("Model Status", table)
        self.assertIn("API key: sk-s********lue", table)
        self.assertIn("Rate limits: models remaining_tokens=123 reset_tokens=30s", table)
        self.assertIn("Model", table)
        self.assertIn("Ctx", table)
        self.assertIn("deepseek-v4-flash", table)
        self.assertIn("128,000", table)
        self.assertIn("118,877", table)
        self.assertIn("groq-qwen3-32b", table)
        self.assertIn("unknown", table)


if __name__ == "__main__":
    unittest.main()
