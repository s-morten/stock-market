"""
Gemini Free API – Hello World proof of concept.

Sends a simple "Hello World" prompt to the Gemini REST API and prints
the response.  Requires GEMINI_API_KEY to be set in the environment or
in the project .env file.

Usage:
    uv run python scripts/gemini_hello_world.py
"""

import json
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from dotenv import load_dotenv

from reit_dashboard.config import get_gemini_api_key

load_dotenv()

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key={api_key}"
)


def main() -> None:
    """Send a Hello World prompt to Gemini and print the response."""
    api_key = get_gemini_api_key()
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        print("Add it to your .env file:  GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    url = _GEMINI_URL.format(api_key=api_key)
    payload = {
        "contents": [{"parts": [{"text": "Hello World"}]}],
        "generationConfig": {"temperature": 0.7},
    }

    print("Sending 'Hello World' to Gemini API …")
    print(f"Model : gemini-2.0-flash")
    print(f"URL   : {url.split('?')[0]}\n")  # hide API key in output

    for attempt in range(1, 4):
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload)

        if response.status_code == 429:
            wait = 30 * attempt
            print(f"Rate-limited (429). Waiting {wait}s before retry {attempt}/3 …")
            import time; time.sleep(wait)
            continue

        response.raise_for_status()
        break
    else:
        print("ERROR: Still rate-limited after 3 attempts. Try again in a minute.")
        sys.exit(1)

    data = response.json()
    reply = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "(no response text)")
    )

    print("=" * 60)
    print("Gemini response:")
    print("=" * 60)
    print(reply)
    print("=" * 60)
    print(f"\nFull JSON response:\n{json.dumps(data, indent=2)}")


if __name__ == "__main__":
    main()
