import json
import os
import unittest
from email.message import Message
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from ddi_guard.assistant import (AssistantConfigurationError, AssistantRateLimitError,
                                 chat_completion, provider_config)


class AssistantProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "DDI_GUARD_AI_PROVIDER": "openai",
            "DDI_GUARD_AI_API_KEY": "unit-test-secret",
            "DDI_GUARD_AI_MODEL": "unit-test-model",
        }, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_openai_configuration_uses_backend_key_and_fixed_official_host(self):
        config = provider_config()
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "unit-test-model")
        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.api_key, "unit-test-secret")

    def test_compatible_provider_rejects_plaintext_remote_base_url(self):
        os.environ["DDI_GUARD_AI_PROVIDER"] = "openai_compatible"
        os.environ["DDI_GUARD_AI_BASE_URL"] = "http://example.org/v1"
        with self.assertRaises(AssistantConfigurationError):
            provider_config()

    def test_chat_completion_uses_bearer_key_and_parses_provider_answer(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({"choices":[{"message":{"content":"  verified mock answer  "}}]}).encode()
        with patch("ddi_guard.assistant.urlopen", return_value=response) as open_url:
            answer = chat_completion([{"role":"user","content":"question"}])
        self.assertEqual(answer, "verified mock answer")
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-secret")
        self.assertNotIn("unit-test-secret", request.data.decode())

    def test_rate_limit_retries_once_then_returns_safe_error(self):
        headers = Message(); headers["Retry-After"] = "0"
        limited = HTTPError("https://api.openai.com/v1/chat/completions", 429, "rate limited", headers, None)
        with patch("ddi_guard.assistant.urlopen", side_effect=[limited, limited]) as open_url, \
                patch("ddi_guard.assistant.time.sleep") as sleep:
            with self.assertRaises(AssistantRateLimitError):
                chat_completion([{"role":"user","content":"question"}])
        self.assertEqual(open_url.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_short_summary_can_use_small_budget_and_skip_rate_limit_retry(self):
        headers = Message(); headers["Retry-After"] = "12"
        limited = HTTPError("https://api.openai.com/v1/chat/completions", 429,
                            "rate limited", headers, None)
        with patch("ddi_guard.assistant.urlopen", side_effect=limited) as open_url, \
                patch("ddi_guard.assistant.time.sleep") as sleep:
            with self.assertRaises(AssistantRateLimitError) as caught:
                chat_completion([{"role": "user", "content": "short summary"}],
                                timeout_seconds=4.0, max_tokens=160,
                                retry_rate_limit=False)
        self.assertEqual(open_url.call_count, 1)
        self.assertEqual(caught.exception.retry_after_seconds, 12.0)
        sleep.assert_not_called()
        request_body = json.loads(open_url.call_args.args[0].data.decode())
        self.assertEqual(request_body["max_tokens"], 160)


if __name__ == "__main__":
    unittest.main()
