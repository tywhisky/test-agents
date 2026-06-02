from openai import OpenAI


class OpenAICompatibleClient:
    """A tiny wrapper around any OpenAI-compatible chat completion API."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("Calling large language model...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            answer = response.choices[0].message.content
            print("Large language model responded successfully.")
            return answer or ""
        except Exception as error:
            print(f"Error occurred when calling LLM API: {error}")
            return "Error: Error occurred when calling language model service."

