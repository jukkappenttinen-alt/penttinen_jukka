"""OpenAI Responses API streaming example."""

import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    prompt = input("Prompt: ")

    stream = client.responses.create(
        model="gpt-5.6-luna",
        reasoning={"effort": "none"},
        max_output_tokens=1024,
        input=prompt,
        stream=True,
    )

    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
