"""Small API-NFL client. Keep credentials out of files, logs, and URLs."""

import os
import time

from dotenv import load_dotenv
import requests

BASE_URL = "https://v1.american-football.api-sports.io"


class Client:
    def __init__(self, root):
        load_dotenv(root / ".env", override=False)
        key = os.environ.get("API_SPORTS_KEY", "").strip()
        if not key or key == "your_key":
            raise ValueError("Add API_SPORTS_KEY to the project's local .env file first.")
        self.session = requests.Session()
        self.session.headers["x-apisports-key"] = key
        self.calls = 0

    def get(self, endpoint, **params):
        for attempt in range(3):
            if self.calls >= 100:
                raise ValueError("Refresh stopped at its 100-request safety limit.")
            self.calls += 1
            time.sleep(0.25)  # Stay below PRO's 300 requests/minute.
            try:
                response = self.session.get(BASE_URL + endpoint, params=params, timeout=30,
                                            allow_redirects=False)
            except requests.RequestException:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise ValueError(f"Network failure requesting {endpoint}; snapshot not replaced.") from None
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            if not 200 <= response.status_code < 300:
                raise ValueError(f"API-NFL {endpoint}: HTTP {response.status_code}; snapshot not replaced.")
            try:
                body = response.json()
            except ValueError:
                raise ValueError(f"API-NFL {endpoint}: invalid JSON") from None
            if not isinstance(body, dict) or body.get("errors"):
                raise ValueError(f"API-NFL {endpoint}: API error; check coverage, quota and subscription in the dashboard.")
            rows = body.get("response")
            if not isinstance(rows, list) or body.get("results") != len(rows):
                raise ValueError(f"API-NFL {endpoint}: unexpected response or incomplete results")
            paging = body.get("paging") or {}
            if paging.get("total", 1) > 1:
                raise ValueError("Paginated response needs explicit handling; refusing an incomplete snapshot.")
            return rows

    def close(self):
        self.session.close()
