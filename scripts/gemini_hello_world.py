"""
Gemini Free API – Hello World proof of concept (google-genai SDK).

Uses the official ``google-genai`` Python SDK to send a "Hello World"
prompt to the Gemini free-tier API and prints the response.

Requires GEMINI_API_KEY to be set in the environment or .env file.

Usage:
    uv run python scripts/gemini_hello_world.py
"""

import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
from google import genai

from reit_dashboard.config import get_gemini_api_key

load_dotenv()

_MODEL = "gemini-2.5-flash-preview-04-17"


def main() -> None:
    """Send a Hello World prompt to Gemini via the google-genai SDK."""
    api_key = get_gemini_api_key()
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        print("Add it to your .env file:  GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    print(f"Sending 'Hello World' to Gemini API (model: {_MODEL}) …\n")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_MODEL,
        contents="Hello World",
    )

    print("=" * 60)
    print("Gemini response:")
    print("=" * 60)
    print(response.text)
    print("=" * 60)


if __name__ == "__main__":
    main()
