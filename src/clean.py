"""Extract set-piece events from StatsBomb event JSON into tidy CSVs in data/processed/.

Outputs
  corners.csv      one row per corner kick, with delivery type, target zone and the shots/xG it led to
  free_kicks.csv   one row per free kick in the attacking third (direct shots and deliveries)
  shots.csv        every shot, tagged with the set-piece phase it came from
  matches.csv      match list with competition and teams
A set piece "leads to" a shot when the same team shoots in the same StatsBomb possession within
20 seconds of the delivery (first and second phase of the set piece)."""
import json, pathlib
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
COMP = {(2, 27): "Premier League 2015/16", (43, 106): "World Cup 2022", (55, 282): "Euro 2024"}
WINDOW = 20  # seconds after the delivery

def secs(e):
    return (e["period"] - 1) * 10_000 + e["minute"] * 60 + e["second"]  # period offset keeps halves apart

def corner_zone(x, y, corner_y, length):
    """Target zone of a corner delivery, standardised so 'near' is the side the corner was taken from."""
    if x is None:
        return "Unknown"
    yn = y if corner_y < 40 else 80 - y  # distance from the corner's touchline
    if length < 15 or x < 90:
        return "Short"
    if x >= 108 and 18 <= yn < 36:
        return "Near post"
    if x >= 108 and 36 <= yn <= 44:
        return "Central (goal area)"
    if x >= 108 and 44 < yn <= 62:
        return "Far post"
    if 102 <= x < 108 and 18 <= yn <= 62:
        return "Penalty spot area"
    if 90 <= x < 102 and 18 <= yn <= 62:
        return "Edge of box"
    return "Wide / beyond box"

def parse(args):
    mid, comp, home, away = args
    ev = json.loads((RAW / "events" / f"{mid}.json").read_text(encoding="utf-8"))
    shots = []
    for e in ev:
        if e["type"]["name"] != "Shot" or e["period"] == 5:  # skip penalty shootouts
            continue
        s = e["shot"]
        shots.append(dict(match_id=mid, competition=comp, team=e["team"]["name"], player=e["player"]["name"],
                          possession=e["possession"], t=secs(e), minute=e["minute"], x=e["location"][0], y=e["location"][1],
                          xg=s.get("statsbomb_xg", 0.0), goal=int(s["outcome"]["name"] == "Goal"),
                          shot_type=s["type"]["name"], body_part=s.get("body_part", {}).get("name"),
                          play_pattern=e["play_pattern"]["name"], outcome=s["outcome"]["name"]))
    sdf = pd.DataFrame(shots)
    corners, fks = [], []
    for e in ev:
        if e["type"]["name"] == "Pass" and e["pass"].get("type", {}).get("name") in ("Corner", "Free Kick"):
            p, kind = e["pass"], e["pass"]["type"]["name"]
            x0, y0 = e["location"]
            if kind == "Free Kick" and x0 < 80:
                continue
            ex, ey = (p.get("end_location") or [None, None])[:2]
            t = secs(e)
            lk = sdf[(sdf.possession == e["possession"]) & (sdf.team == e["team"]["name"]) & (sdf.t >= t) & (sdf.t <= t + WINDOW)] if len(sdf) else sdf
            row = dict(match_id=mid, competition=comp, team=e["team"]["name"],
                       opponent=away if e["team"]["name"] == home else home, player=e["player"]["name"],
                       minute=e["minute"], x=x0, y=y0, end_x=ex, end_y=ey, length=p.get("length"),
                       height=p.get("height", {}).get("name"), technique=p.get("technique", {}).get("name", "Not recorded"),
                       body_part=p.get("body_part", {}).get("name"), completed=int("outcome" not in p),
                       outcome=p.get("outcome", {}).get("name", "Complete"), cross=int(bool(p.get("cross"))),
                       direct_shot_assist=int(bool(p.get("shot_assist") or p.get("goal_assist"))),
                       shots=len(lk), xg=float(lk.xg.sum()) if len(lk) else 0.0,
                       goals=int(lk.goal.sum()) if len(lk) else 0)
            if kind == "Corner":
                row["zone"] = corner_zone(ex, ey, y0, p.get("length", 0))
                row["side"] = "Right" if y0 > 40 else "Left"  # attacking team view: StatsBomb y=0 is the left touchline
                corners.append(row)
            else:
                in_box = ex is not None and ex >= 102 and 18 <= ey <= 62
                row["fk_type"] = "Delivery into box" if in_box and p.get("height", {}).get("name") == "High Pass" else "Short / other pass"
                fks.append(row)
    for s in shots:
        if s["shot_type"] == "Free Kick":  # direct free-kick shots
            fks.append(dict(match_id=mid, competition=comp, team=s["team"], opponent=away if s["team"] == home else home,
                            player=s["player"], minute=s["minute"], x=s["x"], y=s["y"], end_x=None, end_y=None, length=None,
                            height=None, technique="Direct shot", body_part=s["body_part"], completed=None, outcome=s["outcome"],
                            cross=0, direct_shot_assist=0, shots=1, xg=s["xg"], goals=s["goal"], fk_type="Direct shot"))
    return corners, fks, shots

def main():
    PROC.mkdir(parents=True, exist_ok=True)
    jobs, mrows = [], []
    for (c, s), name in COMP.items():
        for m in json.loads((RAW / "matches" / f"{c}_{s}.json").read_text(encoding="utf-8")):
            h, a = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
            jobs.append((m["match_id"], name, h, a))
            mrows.append(dict(match_id=m["match_id"], competition=name, date=m["match_date"], home=h, away=a,
                              home_score=m["home_score"], away_score=m["away_score"]))
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(parse, jobs, chunksize=8))
    corners = pd.DataFrame([r for c, _, _ in res for r in c])
    fks = pd.DataFrame([r for _, f, _ in res for r in f])
    shots = pd.DataFrame([r for _, _, s in res for r in s]).drop(columns="t")
    shots["phase"] = np.select([shots.shot_type == "Penalty", shots.shot_type == "Free Kick",
                                shots.play_pattern == "From Corner", shots.play_pattern == "From Free Kick",
                                shots.play_pattern == "From Throw In"],
                               ["Penalty", "Direct free kick", "Corner phase", "Indirect free-kick phase", "Throw-in phase"], "Open play / other")
    corners.round(4).to_csv(PROC / "corners.csv", index=False)
    fks.round(4).to_csv(PROC / "free_kicks.csv", index=False)
    shots.round(4).to_csv(PROC / "shots.csv", index=False)
    pd.DataFrame(mrows).to_csv(PROC / "matches.csv", index=False)
    print(f"{len(mrows)} matches | {len(corners):,} corners | {len(fks):,} attacking-third free kicks | {len(shots):,} shots")
    print(corners.groupby("competition").size())

if __name__ == "__main__":
    main()
