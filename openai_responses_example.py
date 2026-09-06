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
        reasoning={"effort": "low"},
        max_output_tokens=512,
        input="Explain what is the OpenAI Responses API in simple terms, max 2 sentences."
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
