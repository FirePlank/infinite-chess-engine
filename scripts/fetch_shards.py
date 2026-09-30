"""Download a run's shard-* artifacts in parallel into <dir>/shard-N/ (what `gh run download`
does one at a time, ~0.5 s each).
    python scripts/fetch_shards.py <owner/repo> <run-id> <dir>"""
import io
import json
import time
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

repo, run, out = sys.argv[1:4]
listing = subprocess.run(["gh", "api", f"repos/{repo}/actions/runs/{run}/artifacts?per_page=100"],
                         capture_output=True, check=True).stdout
arts = [a for a in json.loads(listing)["artifacts"] if a["name"].startswith("shard-") and not a["expired"]]


def fetch(a):
    # Concurrent requests can trip GitHub's secondary limit: back off and retry.
    for attempt in range(6):
        r = subprocess.run(["gh", "api", f"repos/{repo}/actions/artifacts/{a['id']}/zip"], capture_output=True)
        if r.returncode == 0:
            zipfile.ZipFile(io.BytesIO(r.stdout)).extractall(f"{out}/{a['name']}")
            return
        print(f"{a['name']}: {r.stderr.decode(errors='replace').strip()[:160]}; retry", file=sys.stderr)
        time.sleep(2 ** attempt)
    raise SystemExit(f"{a['name']}: download failed")


with ThreadPoolExecutor(8) as pool:
    list(pool.map(fetch, arts))
print(f"{len(arts)} shard artifacts", file=sys.stderr)
if not arts:
    sys.exit(1)
