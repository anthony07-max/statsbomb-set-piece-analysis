"""Streamlit dashboard: set-piece effectiveness from StatsBomb open data. Run: streamlit run app/streamlit_app.py"""
import json, pathlib
import numpy as np, pandas as pd, plotly.express as px, streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
st.set_page_config(page_title="Set-piece analysis", layout="wide")

@st.cache_data
def load():
    c = pd.read_csv(P / "corners.csv")
    c["any_shot"] = (c.shots > 0).astype(int)
    c["end_y_std"] = np.where(c.y > 40, 80 - c.end_y, c.end_y)
    return (c, pd.read_csv(P / "free_kicks.csv"), pd.read_csv(P / "shots.csv"),
            pd.read_csv(P / "goals_by_phase.csv"), json.loads((P / "key_findings.json").read_text()))
c, fk, s, phase, F = load()

st.image(str(ROOT / "assets" / "statsbomb-logo.png"), width=180)
st.title("Set-piece effectiveness: corners and free kicks")
st.caption("Data: StatsBomb open data (Premier League 2015/16, FIFA World Cup 2022, UEFA Euro 2024). "
           "Built by Anthony Adusei-Poku (github.com/anthony07-max).")
comps = st.multiselect("Competitions", sorted(c.competition.unique()), default=sorted(c.competition.unique()))
c, fk, s = c[c.competition.isin(comps)], fk[fk.competition.isin(comps)], s[s.competition.isin(comps)]

k = st.columns(4)
k[0].metric("Corners", f"{len(c):,}")
k[1].metric("Corners leading to a shot", f"{100 * c.any_shot.mean():.1f}%")
k[2].metric("xG per corner", f"{c.xg.mean():.3f}")
k[3].metric("Corners per goal", f"{len(c) / max(c.goals.sum(), 1):.0f}")

t1, t2, t3, t4, t5 = st.tabs(["Delivery zones", "Delivery types", "Teams", "Free kicks", "Goal sources"])
def summ(d, by):
    return d.groupby(by).agg(corners=("xg", "size"), shot_rate_pct=("any_shot", lambda x: 100 * x.mean()),
                             xg_per_corner=("xg", "mean"), goals=("goals", "sum")).round(3).reset_index()
with t1:
    z = summ(c, "zone")
    st.plotly_chart(px.bar(z.sort_values("xg_per_corner"), x="xg_per_corner", y="zone", orientation="h",
                           hover_data=["corners", "shot_rate_pct", "goals"], title="xG per corner by delivery zone"), width="stretch")
    team = st.selectbox("Show delivery map for team", ["All teams"] + sorted(c.team.unique()))
    d = c if team == "All teams" else c[c.team == team]
    d = d[d.end_x.notna()]
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_color="#333")
    fig, ax = pitch.draw(figsize=(6, 5))
    pitch.scatter(d.end_x, d.end_y_std, ax=ax, s=8, alpha=.35, color="#1F4E79")
    sh = d[d.any_shot == 1]
    pitch.scatter(sh.end_x, sh.end_y_std, ax=ax, s=12, alpha=.6, color="#D7263D")
    ax.set_title(f"{team}: corner end locations (red = led to a shot), taken from the left", fontsize=9)
    st.pyplot(fig, width="content")
with t2:
    c2 = c.assign(delivery=np.where(c.zone == "Short", "Short corner", c.technique))
    st.dataframe(summ(c2, ["delivery", "height"]), width="stretch", hide_index=True)
with t3:
    att = c.groupby("team").agg(corners=("xg", "size"), xg_per_corner_for=("xg", "mean"), goals_for=("goals", "sum"))
    dfn = c.groupby("opponent").agg(xg_per_corner_against=("xg", "mean"), goals_against=("goals", "sum"))
    t = att.join(dfn).reset_index()
    mn = st.slider("Minimum corners", 5, 150, 20)
    t = t[t.corners >= mn]
    st.plotly_chart(px.scatter(t, x="xg_per_corner_for", y="xg_per_corner_against", size="corners", hover_name="team",
                               title="Corner attack vs defence (lower y = better defence)"), width="stretch")
    st.dataframe(t.sort_values("xg_per_corner_for", ascending=False).round(4), width="stretch", hide_index=True)
with t4:
    st.dataframe(fk.groupby("fk_type").agg(n=("xg", "size"), xg_per_fk=("xg", "mean"), goals=("goals", "sum")).round(4),
                 width="stretch")
    d = fk[fk.fk_type == "Direct shot"]
    st.plotly_chart(px.scatter(d, x="y", y="x", size="xg", color=d.goals.map({1: "Goal", 0: "No goal"}), hover_data=["player", "team"],
                               range_y=[60, 120], title="Direct free-kick shots (x = width, y = distance up the pitch)"), width="stretch")
with t5:
    st.plotly_chart(px.bar(phase[phase.competition.isin(comps)], x="goal_share_pct", y="competition", color="phase", orientation="h",
                           title="Share of goals by phase of play"), width="stretch")
