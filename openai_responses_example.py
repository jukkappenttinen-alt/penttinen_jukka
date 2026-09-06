"""OpenAI Responses API example.

The OpenAI Responses API is a newer generation of the OpenAI chat/completion
interface that lets you send a single prompt and receive a model response using
modern features like reasoning, tool calling, and structured output handling.
It is designed to make building AI-powered applications simpler and more
flexible than older request patterns.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-5.6-luna",
        input="Explain what the OpenAI Responses API is in one short paragraph.",
        reasoning={"effort": "minimal"},
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
