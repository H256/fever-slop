import threading
import time
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import dspy
import httpx
from openai import APIConnectionError

from feverslop.adapters.llm_client import LocalOpenAIClient
from feverslop.llm_concurrency import LLMConcurrencyLimiter, LimitedDspyLM
from feverslop.prompting.dspy_runtime import DspyRuntime, H3SignatureBundle


def _response(content="ok"):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class BlockingBackend:
    def __init__(self, delay=0.05):
        self.delay = delay
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_observed = 0

    def run(self, result):
        with self._lock:
            self.in_flight += 1
            self.max_observed = max(self.max_observed, self.in_flight)
        try:
            time.sleep(self.delay)
            return result
        finally:
            with self._lock:
                self.in_flight -= 1


class LimitedDspyLMAttrLookupTests(unittest.TestCase):
    """Issue #1189: __getattr__ must not recurse when _lm is not set yet."""

    def _fresh_proxy(self):
        # Bypass __init__ so _lm is not set yet (attribute lookups during or
        # before construction must degrade to AttributeError, not recursion).
        return LimitedDspyLM.__new__(LimitedDspyLM)

    def test_fresh_proxy_missing_attr_raises_attribute_error_not_recursion(self):
        proxy = self._fresh_proxy()
        with self.assertRaises(AttributeError):
            proxy.anything

    def test_fresh_proxy_backing_lm_lookup_raises_attribute_error_not_recursion(self):
        proxy = self._fresh_proxy()
        with self.assertRaises(AttributeError):
            proxy._lm

    def test_fresh_proxy_explicit_dunder_lookup_does_not_recurse(self):
        proxy = self._fresh_proxy()
        # Dunders resolve on the type, so this must not raise RecursionError.
        str(proxy)

    def test_fully_constructed_proxy_still_delegates_to_underlying(self):
        underlying = dspy.LM("openai/fake-model", api_key="test", api_base="http://localhost")
        proxy = LimitedDspyLM(underlying, LLMConcurrencyLimiter())

        self.assertIs(proxy.num_retries, underlying.num_retries)
        self.assertIs(proxy.__getattr__("num_retries"), underlying.num_retries)

    def test_fully_constructed_proxy_missing_attr_raises_attribute_error_not_recursion(self):
        class FakeLM:
            def __call__(self, **kwargs):
                return ["ok"]

        proxy = LimitedDspyLM(FakeLM(), LLMConcurrencyLimiter())
        with self.assertRaises(AttributeError):
            proxy.does_not_exist


class LLMConcurrencyTests(unittest.TestCase):
    def test_limited_dspy_lm_is_accepted_by_dspy_context(self):
        import dspy

        from feverslop.llm_concurrency import LimitedDspyLM, LLMConcurrencyLimiter

        underlying = dspy.LM("openai/fake-model", api_key="test", api_base="http://localhost")
        limited = LimitedDspyLM(underlying, LLMConcurrencyLimiter())

        self.assertIsInstance(limited, dspy.BaseLM)
        with dspy.context(lm=limited):
            self.assertIs(dspy.settings.lm, limited)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_direct_openai_calls_default_to_one_in_flight_request(self, mock_openai):
        backend = BlockingBackend()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kwargs: backend.run(_response("direct"))
        mock_openai.return_value = mock_client
        client = LocalOpenAIClient(api_key="test-key")
        start = threading.Barrier(3)

        def call():
            start.wait()
            self.assertEqual("direct", client.complete_prompt("system", "prompt"))

        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(1, backend.max_observed)
        self.assertEqual(1, client.llm_concurrency_snapshot().max_observed)

    @patch("feverslop.adapters.llm_client.OpenAI")
    @patch("feverslop.adapters.llm_client.time.sleep")
    def test_retry_delay_stays_inside_the_concurrency_boundary(self, mock_sleep, mock_openai):
        retry_sleep_entered = threading.Event()
        allow_retry_sleep = threading.Event()

        def sleep_inside_retry(delay):
            retry_sleep_entered.set()
            allow_retry_sleep.wait()

        mock_sleep.side_effect = sleep_inside_retry
        mock_client = MagicMock()
        started = []
        started_lock = threading.Lock()

        def create(**kwargs):
            with started_lock:
                started.append(threading.current_thread().name)
                attempt = len(started)
            if attempt == 1:
                raise APIConnectionError(
                    message="connect failed",
                    request=httpx.Request("GET", "http://localhost"),
                )
            return _response(threading.current_thread().name)

        mock_client.chat.completions.create.side_effect = create
        mock_openai.return_value = mock_client
        client = LocalOpenAIClient(api_key="test-key", max_retries=2, retry_base_delay=0.01)

        first = threading.Thread(
            target=lambda: client.complete_prompt("system", "first"),
            name="first",
        )
        first.start()
        self.assertTrue(retry_sleep_entered.wait(timeout=2))

        second = threading.Thread(
            target=lambda: client.complete_prompt("system", "second"),
            name="second",
        )
        second.start()
        threading.Event().wait(0.05)

        with started_lock:
            self.assertEqual(["first"], started)

        allow_retry_sleep.set()
        first.join(timeout=2)
        second.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_direct_and_dspy_litellm_calls_share_the_same_budget(self, mock_openai):
        backend = BlockingBackend()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda **kwargs: backend.run(_response("direct"))
        mock_openai.return_value = mock_client
        client = LocalOpenAIClient(api_key="test-key")

        class FakeLM:
            def __call__(self, **kwargs):
                return backend.run(["dspy"])

        runtime = DspyRuntime(
            signatures=H3SignatureBundle(object, object, object, object),
            lm_factory=lambda *args, **kwargs: FakeLM(),
            predict_factory=lambda signature: signature,
            context_factory=lambda **kwargs: nullcontext(kwargs),
        )
        dspy_lm = runtime.make_lm(client)
        start = threading.Barrier(3)

        def direct_call():
            start.wait()
            self.assertEqual("direct", client.complete_prompt("system", "prompt"))

        def dspy_call():
            start.wait()
            self.assertEqual(["dspy"], dspy_lm(prompt="prompt"))

        threads = [
            threading.Thread(target=direct_call),
            threading.Thread(target=dspy_call),
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(1, backend.max_observed)
        self.assertEqual(1, client.llm_concurrency_snapshot().max_observed)

    @patch("feverslop.adapters.llm_client.OpenAI")
    def test_conflicting_client_concurrency_limits_are_rejected(self, mock_openai):
        mock_openai.return_value = MagicMock()

        LocalOpenAIClient(api_key="test-key", max_concurrent_requests=1)

        with self.assertRaisesRegex(ValueError, "already configured"):
            LocalOpenAIClient(api_key="test-key", max_concurrent_requests=2)


if __name__ == "__main__":
    unittest.main()
