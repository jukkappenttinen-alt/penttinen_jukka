"""Streaming OpenAI Chat Completions API example."""

import os

from dotenv import load_dotenv
from openai import OpenAI


MODEL_NAME = "gpt-5.6-luna"
API_KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
EXIT_COMMAND = "quit"
PROMPT_TEXT = "You: "
ASSISTANT_PREFIX = "Assistant: "
EXIT_MESSAGE = "Goodbye!"
MISSING_API_KEY_MESSAGE = "OPENAI_API_KEY environment variable is not set."


def main() -> None:
    load_dotenv()
    api_key = os.getenv(API_KEY_ENVIRONMENT_VARIABLE)
    if not api_key:
        raise RuntimeError(MISSING_API_KEY_MESSAGE)

    client = OpenAI(api_key=api_key)
    messages = [{"role": SYSTEM_ROLE, "content": SYSTEM_PROMPT}]

    while True:
        user_input = input(PROMPT_TEXT).strip()
        if user_input.lower() == EXIT_COMMAND:
            print(EXIT_MESSAGE)
            break
        if not user_input:
            continue

        messages.append({"role": USER_ROLE, "content": user_input})
        print(ASSISTANT_PREFIX, end="", flush=True)

        response_text = ""
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                response_text += delta

        print()
        messages.append({"role": ASSISTANT_ROLE, "content": response_text})


if __name__ == "__main__":
    main()
