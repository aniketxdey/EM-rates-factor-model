"""HTTP with retries plus a raw-file cache and a provenance manifest."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import RAW_DIR

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
MANIFEST = os.path.join(RAW_DIR, "manifest.json")

_session = requests.Session()
_session.headers.update({"User-Agent": UA})


def get(url: str, headers: dict | None = None, timeout: int = 90,
        retries: int = 4) -> requests.Response:
    last = None
    for i in range(retries):
        try:
            r = _session.get(url, headers=headers or {}, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
            if r.status_code in (400, 401, 403, 404):
                break
        except requests.RequestException as e:
            last = e
        time.sleep(2 * (i + 1))
    raise last


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {}


def record(name: str, df: pd.DataFrame, source: str) -> str:
    """Write df to data/raw/{name}.csv and log provenance."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.csv")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    text = buf.getvalue()
    with open(path, "w") as f:
        f.write(text)
    man = _load_manifest()
    man[name] = {
        "source": source,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
    }
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    return path


def load(name: str, **kw) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run `python -m pipeline.fetch_all` first")
    return pd.read_csv(path, **kw)
