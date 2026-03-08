import asyncio
import unittest

import opsmeter_sdk as opsmeter


class PythonSdkTests(unittest.TestCase):
    def tearDown(self) -> None:
        opsmeter.reset_for_tests()

    def test_init_is_idempotent(self):
        first = opsmeter.init(api_key="key", enabled=False, environment="prod")
        second = opsmeter.init(api_key="key", enabled=False, environment="prod")

        self.assertTrue(first["did_init"])
        self.assertFalse(second["did_init"])
        self.assertIsNone(second["warning"])

    def test_context_sets_and_restores(self):
        self.assertEqual(opsmeter.get_context(), {})

        with opsmeter.context(user_id="u1", endpoint="/api/chat"):
            current = opsmeter.get_context()
            self.assertEqual(current["user_id"], "u1")
            self.assertEqual(current["endpoint"], "/api/chat")

        self.assertEqual(opsmeter.get_context(), {})

    def test_capture_keeps_business_response(self):
        opsmeter.init(api_key="key", enabled=False, environment="prod")

        with opsmeter.context(
            user_id="u_1",
            tenant_id="tenant_a",
            endpoint="/api/chat",
            feature="assistant",
            prompt_version="v12",
            external_request_id="ext_fixed",
        ):
            result = opsmeter.capture_openai_chat_completion(
                lambda: {
                    "id": "req_1",
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
                request={"model": "gpt-4o-mini"},
            )

        self.assertEqual(result["model"], "gpt-4o-mini")
        opsmeter.flush()

    def test_capture_with_result_returns_telemetry_result(self):
        opsmeter.init(api_key="key", enabled=False, environment="prod")

        result = opsmeter.capture_openai_chat_completion_with_result(
            lambda: {
                "id": "req_2",
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            },
            request={"model": "gpt-4o-mini"},
            await_telemetry_response=True,
        )

        self.assertEqual(result["provider_response"]["model"], "gpt-4o-mini")
        self.assertTrue(result["telemetry"]["ok"])
        self.assertEqual(result["telemetry"]["status"], 204)

    def test_async_capture_with_result(self):
        opsmeter.init(api_key="key", enabled=False, environment="prod")

        async def run_case():
            return await opsmeter.capture_openai_chat_completion_async(
                lambda: asyncio.sleep(
                    0,
                    result={
                        "id": "req_3",
                        "model": "gpt-4o-mini",
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    },
                ),
                request={"model": "gpt-4o-mini"},
                await_telemetry_response=True,
            )

        result = asyncio.run(run_case())
        self.assertEqual(result["provider_response"]["model"], "gpt-4o-mini")
        self.assertTrue(result["telemetry"]["ok"])


if __name__ == "__main__":
    unittest.main()
