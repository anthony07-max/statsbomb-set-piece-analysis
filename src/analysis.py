"""Set-piece analysis on StatsBomb open data: summaries, tests, a logistic model and mplsoccer charts.
Run after clean.py. Writes charts/*.png, data/processed/*.csv summaries and data/processed/key_findings.json."""
import json, pathlib, sqlite3, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from mplsoccer import VerticalPitch, Pitch

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, CH = ROOT / "data" / "processed", ROOT / "charts"
CH.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
CREDIT = "Data: StatsBomb open data"
BLUE, RED, GREY = "#1F4E79", "#D7263D", "#9AA5B1"
PL = "Premier League 2015/16"

c = pd.read_csv(PROC / "corners.csv")
fk = pd.read_csv(PROC / "free_kicks.csv")
s = pd.read_csv(PROC / "shots.csv")
mt = pd.read_csv(PROC / "matches.csv")
c["any_shot"] = (c.shots > 0).astype(int)
c["end_y_std"] = np.where(c.y > 40, 80 - c.end_y, c.end_y)  # all corners shown as if taken from the y=0 corner
F = {"matches": len(mt), "corners": len(c), "corners_per_match": round(len(c) / len(mt), 2),
     "fk_attacking_third": len(fk), "shots": len(s), "competitions": sorted(mt.competition.unique())}

# 1. Overall corner value -------------------------------------------------------------
F.update(corner_shot_rate=round(100 * c.any_shot.mean(), 1), corner_xg=round(c.xg.mean(), 4),
         corner_goal_rate=round(100 * c.goals.mean(), 2), corners_per_goal=round(len(c) / c.goals.sum(), 1),
         corner_goals=int(c.goals.sum()))

# 2. Where do goals come from? (SQL over shots, penalty shootouts excluded in clean.py) -----
con = sqlite3.connect(":memory:"); s.to_sql("shots", con); c.to_sql("corners", con)
phase = pd.read_sql("""SELECT competition, phase, COUNT(*) AS shots, SUM(goal) AS goals, ROUND(SUM(xg), 2) AS xg
                       FROM shots GROUP BY competition, phase""", con)
phase["goal_share_pct"] = (100 * phase.goals / phase.groupby("competition").goals.transform("sum")).round(1)
phase.to_csv(PROC / "goals_by_phase.csv", index=False)
SP = ["Corner phase", "Indirect free-kick phase", "Direct free kick"]
np_ = s[s.phase != "Penalty"]
sp_share = np_.groupby("competition").apply(lambda d: 100 * d[d.phase.isin(SP)].goal.sum() / d.goal.sum()).round(1)
F["set_piece_share_of_non_penalty_goals"] = sp_share.to_dict()
F["set_piece_share_all"] = round(100 * np_[np_.phase.isin(SP)].goal.sum() / np_.goal.sum(), 1)
F["corner_share_of_non_penalty_goals_all"] = round(100 * np_[np_.phase == "Corner phase"].goal.sum() / np_.goal.sum(), 1)

# 3. Delivery zone and type --------------------------------------------------------------
def summ(d, by):
    return d.groupby(by).agg(corners=("xg", "size"), shot_rate_pct=("any_shot", lambda x: 100 * x.mean()),
                             xg_per_corner=("xg", "mean"), goals=("goals", "sum"),
                             first_contact_won_pct=("completed", lambda x: 100 * x.mean())).round(3).reset_index()
ZONES = ["Short", "Near post", "Central (goal area)", "Far post", "Penalty spot area", "Edge of box", "Wide / beyond box"]
zone = summ(c, "zone").set_index("zone").reindex(ZONES).reset_index()
zone.to_csv(PROC / "corner_zones.csv", index=False)
c["delivery"] = np.where(c.zone == "Short", "Short corner",
                         np.where(c.technique.isin(["Inswinging", "Outswinging", "Straight"]), c.technique, "Other / unrecorded"))
deliv = summ(c, "delivery")
deliv.to_csv(PROC / "corner_delivery_types.csv", index=False)
F["zones"] = zone.set_index("zone")[["corners", "shot_rate_pct", "xg_per_corner", "goals"]].to_dict(orient="index")
F["delivery"] = deliv.set_index("delivery")[["corners", "shot_rate_pct", "xg_per_corner", "goals"]].to_dict(orient="index")

io = c[c.technique.isin(["Inswinging", "Outswinging"])]
tab = pd.crosstab(io.technique, io.any_shot)
chi2, p_chi, *_ = stats.chi2_contingency(tab)
mw = stats.mannwhitneyu(io[io.technique == "Inswinging"].xg, io[io.technique == "Outswinging"].xg)
F.update(inswing_shot_rate=round(100 * io[io.technique == "Inswinging"].any_shot.mean(), 1),
         outswing_shot_rate=round(100 * io[io.technique == "Outswinging"].any_shot.mean(), 1),
         inswing_xg=round(io[io.technique == "Inswinging"].xg.mean(), 4),
         outswing_xg=round(io[io.technique == "Outswinging"].xg.mean(), 4),
         in_vs_out_chi2_p=float(f"{p_chi:.2g}"), in_vs_out_xg_mannwhitney_p=float(f"{mw.pvalue:.2g}"),
         inswing_first_contact=round(100 * io[io.technique == "Inswinging"].completed.mean(), 1),
         outswing_first_contact=round(100 * io[io.technique == "Outswinging"].completed.mean(), 1))

# 4. Logistic regression: does a corner produce a shot? -----------------------------------
# fitted on non-short corners with a recorded swing (inswinging / outswinging / straight)
lm = c[(c.zone != "Short") & c.technique.isin(["Inswinging", "Outswinging", "Straight"])].copy()
logit = smf.logit("any_shot ~ C(zone, Treatment('Near post')) + C(technique, Treatment('Inswinging')) + C(competition)", data=lm).fit(disp=0)
F["logit_n"] = len(lm)
lr = pd.DataFrame({"odds_ratio": np.exp(logit.params), "p_value": logit.pvalues}).round(4)
lr.to_csv(PROC / "logit_corner_shot.csv")
key = lambda name: [k for k in lr.index if name in k][0]
F.update(logit_far_post_or=lr.loc[key("Far post"), "odds_ratio"], logit_far_post_p=float(f"{lr.loc[key('Far post'), 'p_value']:.2g}"),
         logit_spot_or=lr.loc[key("Penalty spot"), "odds_ratio"], logit_spot_p=float(f"{lr.loc[key('Penalty spot'), 'p_value']:.2g}"),
         logit_outswing_or=lr.loc[key("Outswinging"), "odds_ratio"], logit_outswing_p=float(f"{lr.loc[key('Outswinging'), 'p_value']:.2g}"))

# 5. Corner-phase shots: headers vs feet ------------------------------------------------
cs = s[s.phase == "Corner phase"]
F.update(corner_shots=len(cs), corner_header_share=round(100 * (cs.body_part == "Head").mean(), 1),
         corner_header_conv=round(100 * cs[cs.body_part == "Head"].goal.mean(), 1),
         corner_foot_conv=round(100 * cs[cs.body_part != "Head"].goal.mean(), 1),
         corner_header_xg=round(cs[cs.body_part == "Head"].xg.mean(), 3), corner_foot_xg=round(cs[cs.body_part != "Head"].xg.mean(), 3))

# 6. Premier League 2015/16 teams: attack and defence per corner --------------------------
plc = c[c.competition == PL]
att = plc.groupby("team").agg(corners_for=("xg", "size"), xg_per_corner_for=("xg", "mean"), goals_for=("goals", "sum"))
dfn = plc.groupby("opponent").agg(corners_against=("xg", "size"), xg_per_corner_against=("xg", "mean"), goals_against=("goals", "sum"))
teams = att.join(dfn).rename_axis("team").reset_index()
teams["corner_xg_diff"] = teams.xg_per_corner_for * teams.corners_for - teams.xg_per_corner_against * teams.corners_against
teams = teams.round(4).sort_values("xg_per_corner_for", ascending=False)
teams.to_csv(PROC / "pl_team_corners.csv", index=False)
F.update(pl_best_corner_attack=teams.team.iloc[0], pl_best_corner_attack_xg=round(teams.xg_per_corner_for.iloc[0], 4),
         pl_worst_corner_attack=teams.team.iloc[-1], pl_worst_corner_attack_xg=round(teams.xg_per_corner_for.iloc[-1], 4),
         pl_most_corner_goals=teams.sort_values("goals_for").team.iloc[-1], pl_most_corner_goals_n=int(teams.goals_for.max()),
         pl_leicester_corner_goals=int(teams.set_index("team").loc["Leicester City", "goals_for"]),
         pl_best_corner_defence=teams.sort_values("xg_per_corner_against").team.iloc[0],
         pl_best_corner_defence_xg=round(teams.xg_per_corner_against.min(), 4),
         pl_worst_corner_defence=teams.sort_values("xg_per_corner_against").team.iloc[-1],
         pl_worst_corner_defence_xg=round(teams.xg_per_corner_against.max(), 4))
team_r = stats.spearmanr(teams.corners_for, teams.xg_per_corner_for)
F.update(pl_corners_vs_quality_spearman=round(team_r.statistic, 3), pl_corners_vs_quality_p=round(team_r.pvalue, 3))

# corner takers (all competitions, 40+ corners)
takers = c.groupby(["player", "team"]).agg(corners=("xg", "size"), xg_per_corner=("xg", "mean"), shot_rate_pct=("any_shot", lambda x: 100 * x.mean())).round(4)
takers = takers[takers.corners >= 40].sort_values("xg_per_corner", ascending=False).reset_index()
takers.to_csv(PROC / "corner_takers.csv", index=False)
F.update(top_taker=takers.player.iloc[0], top_taker_team=takers.team.iloc[0], top_taker_xg=round(takers.xg_per_corner.iloc[0], 4),
         top_taker_corners=int(takers.corners.iloc[0]), takers_ranked=len(takers))

# 7. Free kicks --------------------------------------------------------------------------
fks = fk.groupby("fk_type").agg(n=("xg", "size"), shot_rate_pct=("shots", lambda x: 100 * (x > 0).mean()),
                               xg_per_fk=("xg", "mean"), goals=("goals", "sum")).round(4).reset_index()
fks.to_csv(PROC / "free_kick_types.csv", index=False)
d = fk[fk.fk_type == "Direct shot"]
F.update(direct_fk_shots=len(d), direct_fk_goals=int(d.goals.sum()), direct_fk_conv=round(100 * d.goals.mean(), 1),
         direct_fk_xg=round(d.xg.mean(), 3), direct_fk_xg_total=round(d.xg.sum(), 1),
         fk_box_delivery_n=int((fk.fk_type == "Delivery into box").sum()),
         fk_box_delivery_xg=round(fk[fk.fk_type == "Delivery into box"].xg.mean(), 4))
comp = summ(c, "competition"); comp.to_csv(PROC / "corners_by_competition.csv", index=False)
F["by_competition"] = comp.set_index("competition")[["corners", "shot_rate_pct", "xg_per_corner", "goals"]].to_dict(orient="index")

# 8. Charts ------------------------------------------------------------------------------
def foot(fig):
    fig.text(0.99, 0.01, CREDIT, ha="right", va="bottom", fontsize=7, color="grey")

# 01 delivery end-location heatmap (half pitch, bins aligned to the box, six-yard box and goal posts)
XB, YB = [90, 102, 108, 114, 120.01], [0, 18, 30, 36, 44, 50, 62, 80.01]
long_ = c[(c.zone != "Short") & c.end_x.notna() & (c.end_x >= 90)]
pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_color="#333", line_zorder=2, pitch_color="white")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, stat, title, cmap in [(axes[0], "count", "Share of corner deliveries by landing zone", "Blues"),
                              (axes[1], "mean", "xG per corner by landing zone (zones with 25+ corners)", "Reds")]:
    pitch.draw(ax=ax)
    bs = pitch.bin_statistic(long_.end_x, long_.end_y_std, values=long_.xg if stat == "mean" else None,
                             statistic=stat, bins=(XB, YB))
    cnt = pitch.bin_statistic(long_.end_x, long_.end_y_std, statistic="count", bins=(XB, YB))["statistic"]
    if stat == "count":
        bs["statistic"] = bs["statistic"] / len(long_)
    else:
        bs["statistic"] = np.where(cnt >= 25, bs["statistic"], np.nan)
    pitch.heatmap(bs, ax=ax, cmap=cmap, edgecolor="white", alpha=.9)
    for i in range(bs["statistic"].shape[0]):
        for j in range(bs["statistic"].shape[1]):
            v = bs["statistic"][i, j]
            if np.isnan(v) or (stat == "count" and v < .005):
                continue
            pitch.annotate(f"{v:.0%}" if stat == "count" else f"{v:.3f}", xy=(bs["cx"][i, j], bs["cy"][i, j]), ax=ax,
                           ha="center", va="center", fontsize=8, color="black")
    pitch.scatter(120, 0, ax=ax, marker="^", s=120, color="black", zorder=5)
    ax.set_title(title, fontsize=10)
fig.suptitle(f"Corner deliveries, standardised so every corner is taken from the left (black triangle). {len(long_):,} non-short corners",
             fontsize=11)
foot(fig); fig.savefig(CH / "01_corner_delivery_heatmaps.png", bbox_inches="tight"); plt.close(fig)

# 02 zones
fig, ax1 = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(zone))
bars = ax1.bar(x, zone.xg_per_corner, color=BLUE)
ax1.bar_label(bars, labels=[f"{v:.3f}\n(n={n})" for v, n in zip(zone.xg_per_corner, zone.corners)], fontsize=7)
ax1.set_xticks(x); ax1.set_xticklabels(zone.zone, rotation=15, fontsize=8); ax1.set_ylabel("xG per corner")
ax2 = ax1.twinx(); ax2.plot(x, zone.shot_rate_pct, "o-", color=RED); ax2.set_ylabel("% of corners leading to a shot", color=RED)
ax2.spines["right"].set_visible(True); ax2.set_ylim(0, 60); ax1.set_ylim(0, zone.xg_per_corner.max() * 1.3)
ax1.set_title("Corner value by delivery zone (shots within 20s, same possession)"); foot(fig)
fig.tight_layout(); fig.savefig(CH / "02_corner_zone_value.png"); plt.close(fig)

# 03 delivery type
dd = deliv.sort_values("xg_per_corner")
fig, ax = plt.subplots(figsize=(9, 4))
b = ax.barh(dd.delivery, dd.xg_per_corner, color=[RED if v == "Outswinging" else BLUE if v == "Inswinging" else GREY for v in dd.delivery])
ax.bar_label(b, labels=[f"{v:.3f} xG | {r:.0f}% shot | n={n}" for v, r, n in zip(dd.xg_per_corner, dd.shot_rate_pct, dd.corners)], fontsize=8)
ax.set(title="Corner delivery type: xG and shot rate per corner", xlabel="xG per corner", xlim=(0, dd.xg_per_corner.max() * 1.7))
foot(fig); fig.tight_layout(); fig.savefig(CH / "03_corner_delivery_types.png"); plt.close(fig)

# 04 corner-phase shot map
pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_color="#444", pad_bottom=-5)
fig, ax = pitch.draw(figsize=(7, 6.5))
ng, gl = cs[cs.goal == 0], cs[cs.goal == 1]
pitch.scatter(ng.x, ng.y, s=ng.xg * 600 + 8, ax=ax, color=GREY, alpha=.45, edgecolor="none", label="No goal")
pitch.scatter(gl.x, gl.y, s=gl.xg * 600 + 8, ax=ax, color=RED, edgecolor="black", lw=.5, label="Goal")
ax.legend(loc="lower left", fontsize=8)
ax.set_title(f"All {len(cs):,} shots from corner phases (size = xG)\n{F['corner_header_share']}% are headers", fontsize=11)
foot(fig); fig.savefig(CH / "04_corner_shot_map.png", bbox_inches="tight"); plt.close(fig)

# 05 PL team scatter
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.scatter(teams.xg_per_corner_for, teams.xg_per_corner_against, s=teams.corners_for * 1.2, color=BLUE, alpha=.6)
for _, r in teams.iterrows():
    ax.annotate(r.team.replace(" United", " Utd").replace("AFC ", ""), (r.xg_per_corner_for, r.xg_per_corner_against), fontsize=7,
                xytext=(4, 3), textcoords="offset points")
ax.axvline(plc.xg.mean(), color="grey", ls=":"); ax.axhline(plc.xg.mean(), color="grey", ls=":")
ax.invert_yaxis()
ax.set(title="Premier League 2015/16: corner attack vs corner defence\n(top right = good at both; bubble = corners won)",
       xlabel="xG created per corner won", ylabel="xG conceded per corner faced (inverted)")
foot(fig); fig.tight_layout(); fig.savefig(CH / "05_pl_team_corner_attack_defence.png"); plt.close(fig)

# 06 goal share by phase
pv = phase.pivot(index="competition", columns="phase", values="goal_share_pct").fillna(0)
cols = ["Open play / other", "Throw-in phase", "Corner phase", "Indirect free-kick phase", "Direct free kick", "Penalty"]
colors = ["#C9D1D9", "#8FA3B5", RED, "#F28E2B", "#FFBE0B", "#555555"]
fig, ax = plt.subplots(figsize=(10, 3.8)); left = np.zeros(len(pv))
for col, colr in zip(cols, colors):
    v = pv.get(col, pd.Series(0, index=pv.index)).values
    ax.barh(pv.index, v, left=left, color=colr, label=col)
    for i, (l, w) in enumerate(zip(left, v)):
        if w >= 4: ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8)
    left += v
ax.set(title="Where goals come from: share of goals by phase of play", xlabel="% of goals (shootouts excluded)", xlim=(0, 100))
ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2)); foot(fig)
fig.tight_layout(); fig.savefig(CH / "06_goal_share_by_phase.png"); plt.close(fig)

# 07 direct free kicks
pitch = VerticalPitch(pitch_type="statsbomb", half=True, line_color="#444", pad_bottom=-5)
fig, ax = pitch.draw(figsize=(7, 6.5))
dn, dg = d[d.goals == 0], d[d.goals == 1]
pitch.scatter(dn.x, dn.y, s=dn.xg * 1500 + 10, ax=ax, color=GREY, alpha=.5, edgecolor="none", label="No goal")
pitch.scatter(dg.x, dg.y, s=dg.xg * 1500 + 10, ax=ax, color=RED, edgecolor="black", lw=.5, label="Goal")
ax.legend(loc="lower left", fontsize=8)
ax.set_title(f"Direct free-kick shots: {F['direct_fk_goals']} goals from {F['direct_fk_shots']} attempts ({F['direct_fk_conv']}%)\n"
             f"mean xG {F['direct_fk_xg']} (size = xG)", fontsize=11)
foot(fig); fig.savefig(CH / "07_direct_free_kicks.png", bbox_inches="tight"); plt.close(fig)

# 08 corner takers
tk = takers.head(15).sort_values("xg_per_corner")
fig, ax = plt.subplots(figsize=(9, 5.5))
b = ax.barh(tk.player + " (" + tk.team + ")", tk.xg_per_corner, color=BLUE)
ax.bar_label(b, labels=[f"{v:.3f} (n={n})" for v, n in zip(tk.xg_per_corner, tk.corners)], fontsize=7)
ax.axvline(c.xg.mean(), color=RED, ls="--", label=f"All corners {c.xg.mean():.3f}")
ax.set(title="Top 15 corner takers by xG created per corner (40+ corners)", xlabel="xG per corner")
ax.tick_params(axis="y", labelsize=8); ax.legend(fontsize=8); foot(fig)
fig.tight_layout(); fig.savefig(CH / "08_corner_takers.png"); plt.close(fig)

(PROC / "key_findings.json").write_text(json.dumps(F, indent=2, default=str))
print(json.dumps(F, indent=2, default=str))
print(zone.to_string()); print(deliv.to_string()); print(fks.to_string()); print(lr.to_string())
print(teams.to_string()); print(takers.head(10).to_string()); print(phase.to_string())
