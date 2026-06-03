"""Shared test configuration.

Some library modules construct API clients at import time, which requires
credentials to be present. Provide dummy values so the modules can be imported
offline — no network calls are made by the unit tests themselves.
"""

import os

os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.invalid/")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
