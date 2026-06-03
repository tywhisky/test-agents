import unittest

from agent_playground.llm import HelloAgentsLLM


class FakeDelta:
    content = "hello"


class FakeChoice:
    delta = FakeDelta()


class FakeChunk:
    choices = [FakeChoice()]


class FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return [FakeChunk()]


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class HelloAgentsLLMTests(unittest.TestCase):
    def test_think_collects_streamed_content(self):
        client = HelloAgentsLLM(
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
        client.client = FakeClient()

        answer = client.think([{"role": "user", "content": "hi"}])

        self.assertEqual(answer, "hello")
        self.assertTrue(client.client.chat.completions.kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
