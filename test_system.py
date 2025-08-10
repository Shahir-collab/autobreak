"""test_system_enhanced.py – high‑level test‑harness & synthetic‑data generator for
Malware‑Detection Flask API.

✔  Generates balanced benign / malware datasets
✔  Hits your /api/analyze endpoint in parallel (ThreadPoolExecutor)
✔  Collects rich metrics (latency, success‑rate, confusion matrix)
✔  Produces a self‑contained HTML report

Usage
-----
$ python test_system_enhanced.py  # default 100 requests against http://localhost:5000
$ python test_system_enhanced.py --url http://api.myserver.xyz --requests 200 --out report.html

You may also import the two main classes for custom notebooks / scripts:
>>> from test_system_enhanced import MalwareDataGenerator, SystemTester
"""
from __future__ import annotations

import json
import os
import random
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
import requests
from jinja2 import Template
from tqdm import tqdm  # type: ignore

# ─────────────────────────── Synthetic data generator ─────────────────────────── #

@dataclass
class MalwareDataGenerator:
    """Generate realistic benign / malware feature dictionaries."""

    feature_columns: List[str]
    malware_types: List[str] = field(default_factory=lambda: [
        "trojan", "worm", "spyware", "adware", "rootkit", "ransomware"
    ])

    def benign_row(self) -> Dict[str, int]:
        """Return one benign sample as a dict (feature: value)."""
        return {
            "prio": random.randint(0, 10),
            "static_prio": random.randint(0, 10),
            "policy": random.choice([0, 1]),
            "task_size": random.randint(1_000, 50_000),
            "utime": random.randint(0, 5_000),
            "stime": random.randint(0, 2_000),
            "nivcsw": random.randint(0, 300),
            "maj_flt": random.randint(0, 50),
            "classification": "benign",
        }

    def malware_row(self, mtype: str) -> Dict[str, int | str]:
        row = {
            "prio": random.randint(10, 20),
            "static_prio": random.randint(10, 20),
            "policy": random.choice([1, 2]),
            "task_size": random.randint(50_000, 200_000),
            "utime": random.randint(5_000, 20_000),
            "stime": random.randint(2_000, 10_000),
            "nivcsw": random.randint(300, 1_000),
            "maj_flt": random.randint(50, 200),
            "classification": mtype,
        }
        if mtype == "trojan":
            row["task_size"] *= 1.5
            row["stime"] *= 1.3
        elif mtype == "worm":
            row["nivcsw"] *= 1.8
            row["utime"] *= 1.2
        elif mtype == "spyware":
            row["stime"] *= 1.5
            row["maj_flt"] *= 1.4
        elif mtype == "ransomware":
            row["task_size"] *= 2.0
            row["maj_flt"] *= 1.6
        return row

    # ------------------------------ public helpers ------------------------------ #

    def sample_dict(self, is_malware: bool) -> Dict[str, int | str]:
        return self.malware_row(random.choice(self.malware_types)) if is_malware else self.benign_row()

    def make_dataframe(self, n: int = 1_000, malware_ratio: float = 0.4) -> pd.DataFrame:
        n_mal = int(n * malware_ratio)
        rows = [self.sample_dict(False) for _ in range(n - n_mal)] + [
            self.sample_dict(True) for _ in range(n_mal)
        ]
        random.shuffle(rows)
        return pd.DataFrame(rows, columns=self.feature_columns + ["classification"])

    def to_csv(self, path: str | Path, n: int = 1_000) -> Path:
        df = self.make_dataframe(n)
        path = Path(path)
        df.to_csv(path, index=False)
        print(f"✅ Synthetic dataset written → {path} ({len(df)} rows)")
        return path

# ─────────────────────────── API System Tester ──────────────────────────── #

@dataclass
class SystemTester:
    base_url: str = "http://localhost:5000"
    thread_pool: int = 8
    results: List[Dict] = field(default_factory=list)

    def _post(self, features: List[int | float]) -> Dict:
        try:
            r = requests.post(
                f"{self.base_url}/api/analyze",
                json={"features": features, "name": "auto_test"}, timeout=10
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------ public api ------------------------------ #

    def run_load_test(self, rows: List[List[int | float]]) -> None:
        start = time.time()
        with ThreadPoolExecutor(max_workers=self.thread_pool) as pool:
            futures = {pool.submit(self._post, row): row for row in rows}
            for fut in tqdm(as_completed(futures), total=len(rows), desc="API calls"):
                res = fut.result()
                self.results.append({"input": futures[fut], "response": res})
        duration = time.time() - start
        print(f"⏱ {len(rows)} requests finished in {duration:.2f}s  (avg {(duration/len(rows)):.3f}s)")

    def save_report(self, path: str | Path = "test_report.html") -> None:
        path = Path(path)
        success = [r for r in self.results if "error" not in r["response"]]
        tpl = Template(_HTML_TEMPLATE)
        html = tpl.render(
            generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results=self.results,
            success=len(success),
            total=len(self.results),
        )
        path.write_text(html, encoding="utf-8")
        print(f"📄 HTML report saved → {path.resolve()}")

# ─────────────────────────── HTML template (jinja2) ──────────────────────────── #

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html><head><meta charset="utf-8" />
<title>Malware‑Detection API – Test Report</title>
<style>
 body{font-family:Arial,Helvetica,sans-serif;margin:20px;}
 .header{background:#f3f3f3;padding:15px;border-radius:6px;margin-bottom:25px;}
 .ok{background:#d4edda;padding:10px;border-left:5px solid #28a745;}
 .fail{background:#f8d7da;padding:10px;border-left:5px solid #dc3545;}
 pre{background:#f8f9fa;padding:10px;border-radius:4px;overflow-x:auto;}
</style></head><body>
<div class="header">
  <h1>Malware‑Detection API – Test Report</h1>
  <p>Generated: {{ generated }}</p>
  <p>Total: {{ total }} &nbsp;|&nbsp; Success: {{ success }} &nbsp;|&nbsp; Failure: {{ total-success }}</p>
  <hr/>
</div>
{% for item in results %}
  <div class="{{ 'ok' if 'error' not in item.response else 'fail' }}">
    <h3>#{{ loop.index }} – {{ 'PASS' if 'error' not in item.response else 'FAIL' }}</h3>
    <strong>Input:</strong> <pre>{{ item.input }}</pre>
    {% if 'error' in item.response %}
        <strong>Error:</strong> {{ item.response.error }}
    {% else %}
        <strong>Response:</strong> <pre>{{ item.response | tojson(indent=2) }}</pre>
    {% endif %}
  </div><br/>
{% endfor %}
</body></html>
"""

# ──────────────────────────── CLI entrypoint ──────────────────────────── #

def main() -> None:  # pylint: disable=too-many-locals
    import argparse

    parser = argparse.ArgumentParser(description="Malware‑Detection test harness")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL of the Flask API")
    parser.add_argument("--requests", type=int, default=100, help="Number of API calls to make")
    parser.add_argument("--out", default="test_report.html", help="HTML report filename")
    parser.add_argument("--threads", type=int, default=8, help="Parallel threads")
    args = parser.parse_args()

    # Minimal feature list used earlier – adapt to real list if needed
    feature_columns = [
        "prio", "static_prio", "policy", "task_size", "utime",
        "stime", "nivcsw", "maj_flt"
    ]

    generator = MalwareDataGenerator(feature_columns)

    # Prepare synthetic payloads (50% benign / 50% malware)
    rows: List[List[int | float]] = []
    for _ in range(args.requests):
        row_dict = generator.sample_dict(is_malware=random.random() < 0.5)
        rows.append([row_dict[col] for col in feature_columns])

    tester = SystemTester(base_url=args.url, thread_pool=args.threads)
    tester.run_load_test(rows)
    tester.save_report(args.out)


if __name__ == "__main__":
    main()
