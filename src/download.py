"""Download StatsBomb Open Data (event JSON) for a subset of competitions into data/raw/.

Source: StatsBomb Open Data, https://github.com/statsbomb/open-data
Data provided by StatsBomb (Hudl StatsBomb) under their open data user agreement
(https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf). Raw files are not
committed to this repo.

Subset (kept small on purpose):
  Premier League 2015/16 (competition 2, season 27), a full 380-match league season
  FIFA World Cup 2022    (competition 43, season 106)
  UEFA Euro 2024         (competition 55, season 282)
"""
import json, pathlib, requests
from concurrent.futures import ThreadPoolExecutor

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPS = [(2, 27), (43, 106), (55, 282)]
S = requests.Session()

def get(url, out):
    if out.exists() and out.stat().st_size > 0:
        return
    r = S.get(url, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)

def main():
    (RAW / "matches").mkdir(parents=True, exist_ok=True)
    (RAW / "events").mkdir(parents=True, exist_ok=True)
    get(f"{BASE}/competitions.json", RAW / "competitions.json")
    ids = []
    for c, s in COMPS:
        out = RAW / "matches" / f"{c}_{s}.json"
        get(f"{BASE}/matches/{c}/{s}.json", out)
        m = json.loads(out.read_text())
        print(f"competition {c} season {s}: {len(m)} matches")
        ids += [x["match_id"] for x in m]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: get(f"{BASE}/events/{i}.json", RAW / "events" / f"{i}.json"), ids))
    print("Event files:", len(list((RAW / "events").glob("*.json"))))

if __name__ == "__main__":
    main()
