"""
Gemini API client for LLM-based property count extraction.

Uses the official ``google-genai`` Python SDK (free tier, gemini-2.5-flash)
to determine the total number of properties reported in a REIT's Item 2
filing tables.

Rate limits (free tier, as of 2025)
------------------------------------
- gemini-2.5-flash: 10 RPM / 250 RPD
- A 7-second sleep between calls keeps us well inside the limit.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

# Default model – free tier, fast, sufficient for structured extraction.
DEFAULT_MODEL = "gemini-2.5-flash"

# Seconds to wait between consecutive Gemini requests to respect 10 RPM.
_GEMINI_REQUEST_DELAY = 7.0

# Retry settings for quota-exhausted (429) responses.
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 30.0  # seconds; doubles on each retry

# Prompt template; {tables_text} is substituted before the API call.
_PROMPT_TEMPLATE = """\
You are analyzing tables extracted from Item 2 "Properties" of a REIT \
(Real Estate Investment Trust) annual 10-K SEC filing.

Your task: determine the TOTAL number of individual real estate properties \
(buildings, facilities, locations, or assets) owned, leased, or managed by \
this company as reported in these tables.

Rules:
- Look for explicit totals labelled "Total", "Grand total", or similar.
- If the table lists individual locations with counts, sum them up.
- Ignore percentages, dollar values, and square-footage unless the value \
clearly represents a count of distinct properties.
- If you cannot determine a clear total property count, return null.

Return ONLY valid JSON with no markdown fences:
{{"total_properties": <integer or null>, "notes": "<one concise sentence>"}}

Tables from Item 2:
---
{tables_text}
---"""


class GeminiPropertyExtractor:
    """
    Extracts total property counts from Item 2 table text via Gemini.

    Uses the official ``google-genai`` SDK.

    Parameters:
        api_key: Google Gemini API key.
        model:   Gemini model name (default: ``gemini-2.5-flash``).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def extract_property_count(
        self,
        tables_text: str,
    ) -> dict[str, Any]:
        """
        Send *tables_text* to Gemini and return the parsed property count.

        A fixed delay of :data:`_GEMINI_REQUEST_DELAY` seconds is observed
        before each API call so the caller stays inside the free-tier RPM
        limit even when processing many filings in a loop.

        Parameters:
            tables_text: Plain-text TSV tables extracted from Item 2.

        Returns:
            dict with keys:
                ``total_properties`` (int or None) – extracted count,
                ``notes`` (str) – Gemini's explanation.
        """
        if not tables_text.strip():
            logger.info("Gemini: empty tables_text – skipping API call.")
            return {"total_properties": None, "notes": "No table text provided."}

        truncated_text = tables_text[:12_000]
        prompt = _PROMPT_TEMPLATE.format(tables_text=truncated_text)

        _SEPARATOR = "=" * 72
        logger.debug(
            "\n%s\n[GEMINI] EXTRACTED TABLES TEXT (%d chars, truncated to %d)\n%s\n%s\n%s",
            _SEPARATOR, len(tables_text), len(truncated_text),
            _SEPARATOR, truncated_text, _SEPARATOR,
        )
        logger.debug(
            "\n%s\n[GEMINI] FULL PROMPT SENT TO %s (%d chars)\n%s\n%s\n%s",
            _SEPARATOR, self._model, len(prompt), _SEPARATOR, prompt, _SEPARATOR,
        )

        print(f"\n{'=' * 72}")
        print(f"[GEMINI] EXTRACTED TABLES TEXT ({len(tables_text)} chars, truncated to {len(truncated_text)})")
        print("=" * 72)
        print(truncated_text)
        print("=" * 72)
        print(f"\n[GEMINI] FULL PROMPT SENT TO {self._model} ({len(prompt)} chars)")
        print("=" * 72)
        print(prompt)
        print("=" * 72 + "\n")

        logger.debug("[GEMINI] Sleeping %.1fs before API call (rate-limit guard).", _GEMINI_REQUEST_DELAY)
        time.sleep(_GEMINI_REQUEST_DELAY)

        raw_text = self._generate_with_retry(prompt)

        logger.debug("\n%s\n[GEMINI] RAW RESPONSE TEXT\n%s\n%s\n%s",
                     _SEPARATOR, _SEPARATOR, raw_text, _SEPARATOR)
        print(f"\n[GEMINI] RAW RESPONSE\n{'=' * 72}")
        print(raw_text)
        print("=" * 72 + "\n")

        # Strip accidental markdown fences that some models add.
        raw_text = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

        try:
            result: dict[str, Any] = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.warning("Gemini returned non-JSON: %r – %s", raw_text[:200], exc)
            print(f"[GEMINI] WARNING: non-JSON response: {raw_text[:200]!r}")
            return {"total_properties": None, "notes": f"Parse error: {exc}"}

        # Normalise total_properties to int | None.
        raw_count = result.get("total_properties")
        if raw_count is not None:
            try:
                result["total_properties"] = int(raw_count)
            except (TypeError, ValueError):
                result["total_properties"] = None

        logger.info("[GEMINI] Result: total_properties=%s  notes=%r",
                    result.get("total_properties"), result.get("notes"))
        print(f"[GEMINI] Result → total_properties={result.get('total_properties')}  "
              f"notes={result.get('notes')!r}\n")

        return result

    def _generate_with_retry(self, prompt: str) -> str:
        """
        Call the Gemini SDK with exponential back-off on quota errors (429).

        Parameters:
            prompt: Full prompt string to send.

        Returns:
            Raw response text from the model.

        Raises:
            google.genai.errors.ClientError: If all retries are exhausted.
        """
        delay = _RETRY_BASE_DELAY
        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
                return response.text or "{}"
            except genai_errors.ClientError as exc:
                if exc.code != 429 or attempt > _MAX_RETRIES:
                    raise
                print(
                    f"[GEMINI] 429 quota exceeded on attempt {attempt}/{_MAX_RETRIES}. "
                    f"Waiting {delay:.0f}s before retry…"
                )
                logger.warning("[GEMINI] 429 on attempt %d/%d – sleeping %.0fs.",
                               attempt, _MAX_RETRIES, delay)
                time.sleep(delay)
                delay *= 2

        raise RuntimeError("Retry loop exited unexpectedly.")

