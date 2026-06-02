import unittest

from agent_playground.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_accepts_model_name(self):
        env = {
            "API_KEY": "test-api-key",
            "BASE_URL": "https://example.test/v1",
            "MODEL_NAME": "test-model",
            "TAVILY_API_KEY": "test-tavily-key",
        }

        config = load_config(env)

        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.tavily_api_key, "test-tavily-key")

    def test_load_config_reports_missing_values(self):
        config = load_config({})

        self.assertEqual(
            config.missing_values,
            ["API_KEY", "BASE_URL", "MODEL_ID or MODEL_NAME", "TAVILY_API_KEY"],
        )


if __name__ == "__main__":
    unittest.main()
