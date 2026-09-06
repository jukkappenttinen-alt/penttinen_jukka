"""Generate marketing copy with the OpenRouter Chat Completions API."""

import argparse
import os
from collections.abc import Iterator

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError


MODEL_NAME = "deepseek-v4-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENVIRONMENT_VARIABLE = "OPENROUTER_API_KEY"

DEFAULT_STYLE = "clear and persuasive"
DEFAULT_TEXT_LENGTH = 500
DEFAULT_VERSION_COUNT = 3
DEFAULT_TEMPERATURE = 1.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0

MIN_TEXT_LENGTH = 1
MIN_VERSION_COUNT = 1
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_PENALTY = -2.0
MAX_PENALTY = 2.0

SYSTEM_PROMPT = (
    "You are an expert marketing copywriter. Create polished, original marketing "
    "material that follows the user's requested subject, style, length, and number "
    "of versions. Label every version clearly. Keep each version close to the "
    "requested character count, including spaces."
)
USER_PROMPT_TEMPLATE = """Create {version_count} different marketing copy versions.

Subject: {subject}
Style: {style}
Target length per version in characters: {text_length}

Return only the numbered marketing copy versions, with no analysis or introduction."""

SYSTEM_ROLE = "system"
USER_ROLE = "user"
STREAM_ENABLED = True
EMPTY_TEXT = ""
OUTPUT_SEPARATOR = "\n"
MARKETING_COPY_LABEL = "Marketing copy:\n"
MISSING_API_KEY_MESSAGE = (
    "OPENROUTER_API_KEY environment variable is not set."
)
API_ERROR_MESSAGE = "The marketing copy request failed: {error}"
INVALID_ARGUMENT_MESSAGE = "{argument} must be {requirement}."
SUBJECT_PROMPT = "Subject: "
STYLE_PROMPT = "Style [{default}]: "
TEXT_LENGTH_PROMPT = "Text length in characters [{default}]: "
VERSION_COUNT_PROMPT = "Number of versions [{default}]: "


def parse_arguments() -> argparse.Namespace:
    """Parse optional model controls from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate marketing material with DeepSeek through OpenRouter."
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=DEFAULT_FREQUENCY_PENALTY,
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=DEFAULT_PRESENCE_PENALTY,
    )
    return parser.parse_args()


def ask_for_content_attributes(arguments: argparse.Namespace) -> None:
    """Ask the user for all marketing-copy attributes."""
    arguments.subject = input(SUBJECT_PROMPT).strip()
    arguments.style = input(STYLE_PROMPT.format(default=DEFAULT_STYLE)).strip()
    arguments.text_length = int(
        input(TEXT_LENGTH_PROMPT.format(default=DEFAULT_TEXT_LENGTH)).strip()
        or DEFAULT_TEXT_LENGTH
    )
    arguments.versions = int(
        input(VERSION_COUNT_PROMPT.format(default=DEFAULT_VERSION_COUNT)).strip()
        or DEFAULT_VERSION_COUNT
    )
    if not arguments.style:
        arguments.style = DEFAULT_STYLE


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject values that cannot produce a useful request."""
    if not arguments.subject.strip():
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="subject", requirement="non-empty"))
    if arguments.text_length < MIN_TEXT_LENGTH:
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="length", requirement=f"at least {MIN_TEXT_LENGTH}"))
    if arguments.versions < MIN_VERSION_COUNT:
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="versions", requirement=f"at least {MIN_VERSION_COUNT}"))
    if not MIN_TEMPERATURE <= arguments.temperature <= MAX_TEMPERATURE:
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="temperature", requirement=f"between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"))
    if not MIN_TOP_P <= arguments.top_p <= MAX_TOP_P:
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="top-p", requirement=f"between {MIN_TOP_P} and {MAX_TOP_P}"))
    if arguments.max_output_tokens < MIN_TEXT_LENGTH:
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="max-output-tokens", requirement=f"at least {MIN_TEXT_LENGTH}"))
    if not MIN_PENALTY <= arguments.frequency_penalty <= MAX_PENALTY:
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="frequency-penalty", requirement=f"between {MIN_PENALTY} and {MAX_PENALTY}"))
    if not MIN_PENALTY <= arguments.presence_penalty <= MAX_PENALTY:
        raise ValueError(INVALID_ARGUMENT_MESSAGE.format(argument="presence-penalty", requirement=f"between {MIN_PENALTY} and {MAX_PENALTY}"))


def build_user_prompt(arguments: argparse.Namespace) -> str:
    """Build the user message from the requested marketing attributes."""
    return USER_PROMPT_TEMPLATE.format(
        subject=arguments.subject.strip(),
        style=arguments.style.strip(),
        text_length=arguments.text_length,
        version_count=arguments.versions,
    )


def stream_marketing_copy(
    client: OpenAI, arguments: argparse.Namespace
) -> Iterator[str]:
    """Yield generated text chunks from a streamed Chat Completions response."""
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": SYSTEM_ROLE, "content": SYSTEM_PROMPT},
            {"role": USER_ROLE, "content": build_user_prompt(arguments)},
        ],
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        max_tokens=arguments.max_output_tokens,
        frequency_penalty=arguments.frequency_penalty,
        presence_penalty=arguments.presence_penalty,
        stream=STREAM_ENABLED,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def main() -> None:
    """Run the marketing-copy generator."""
    try:
        arguments = parse_arguments()
        ask_for_content_attributes(arguments)
        validate_arguments(arguments)
        load_dotenv()
        api_key = os.getenv(API_KEY_ENVIRONMENT_VARIABLE)
        if not api_key:
            raise RuntimeError(MISSING_API_KEY_MESSAGE)

        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        print(MARKETING_COPY_LABEL, end=EMPTY_TEXT, flush=True)
        for content in stream_marketing_copy(client, arguments):
            print(content, end=EMPTY_TEXT, flush=True)
        print(OUTPUT_SEPARATOR, end=EMPTY_TEXT)
    except (OpenAIError, RuntimeError, ValueError) as error:
        print(API_ERROR_MESSAGE.format(error=error))


if __name__ == "__main__":
    main()