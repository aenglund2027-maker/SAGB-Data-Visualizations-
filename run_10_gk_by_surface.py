"""
Step 10 (Graph 4, part 2): where do the Giant Killers' extra upset wins come from?

Splits each player's Giant Killer Score (actual - expected upset wins) by surface.
Expected wins come from the same season-adjusted ranking + surface baseline
(p_log_rank), so a positive clay score means "more upsets on clay than a player
with that ranking gap would usually get ON CLAY" - the general clay effect is
already in the baseline.

Because the score is a sum over matches, the three surface scores add up exactly
to the overall score. A big surface score can simply mean many matches on that
surface, so opportunities and per-100-match rates are printed next to it.

Input:  data/processed/gk_match_predictions.csv   (one row per underdog opportunity, 2016-2025)
        data/processed/graph4_chart_data.csv      (overall scores, flags, player order)
Output: outputs/figures/graph4b_surface_split.png / .svg
        data/processed/graph4b_surface_data.csv
Run from the project root:  python3 run_10_gk_by_surface.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src import config

pd.set_option("display.width", 200)

SURFACES = ["Hard", "Clay", "Grass"]
TOP_N = 20
MIN_SURFACE_OPPS = 20   # for the "best on each surface" lists only

# Style tokens (same as run_08_figures.py); surface colors follow tennis convention
SURFACE_BG = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SURFACE_COLORS = {"Hard": "#2a78d6", "Clay": "#eb6834", "Grass": "#1baf7a"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE_BG, "axes.facecolor": SURFACE_BG, "savefig.facecolor": SURFACE_BG,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "axes.edgecolor": AXIS, "font.size": 10,
})


def surface_scores(pm):
    """One row per player per surface: opportunities, upset wins, expected wins, score, se."""
    g = pm.groupby(["dog_id", "surface"])
    out = g.agg(opportunities=("upset", "size"), upset_wins=("upset", "sum"),
                expected_wins=("p_log_rank", "sum")).reset_index()
    var = pm.assign(v=pm["p_log_rank"] * (1 - pm["p_log_rank"])).groupby(["dog_id", "surface"])["v"].sum()
    out["se"] = np.sqrt(var.to_numpy())
    out["score"] = out["upset_wins"] - out["expected_wins"]
    out["score_per_100"] = 100 * out["score"] / out["opportunities"]
    return out


def sanity_checks(pm, long, overall):
    print("\n--- Sanity checks ---")
    print(f"Opportunities: {len(pm):,}, seasons {pm['season'].min()}-{pm['season'].max()}")
    print("Surfaces:", pm["surface"].value_counts().to_dict())
    print(f"Missing p_log_rank: {pm['p_log_rank'].isna().sum()}")
    # The surface scores must add back up to the overall Graph 4 score
    summed = long.groupby("dog_id")[["opportunities", "upset_wins", "score"]].sum().add_suffix("_sum")
    chk = overall.set_index("dog_id").join(summed)
    print(f"Max |opportunities - sum over surfaces|: {(chk['opportunities'] - chk['opportunities_sum']).abs().max()}")
    print(f"Max |upset wins - sum over surfaces|:    {(chk['upset_wins'] - chk['upset_wins_sum']).abs().max()}")
    print(f"Max |gk_score - sum over surfaces|:      {(chk['gk_score'] - chk['score_sum']).abs().max():.2e}")
    # League-wide: is any surface systematically under- or over-predicted?
    print("\nAll players, all opportunities, by surface (should be near 0 if the baseline handles surface well):")
    tot = long.groupby("surface")[["opportunities", "upset_wins", "expected_wins", "score"]].sum()
    tot["score_per_100"] = 100 * tot["score"] / tot["opportunities"]
    print(tot.round(2).to_string())


def wide_table(long, overall, top_n):
    top = overall.sort_values("gk_score", ascending=False).head(top_n)
    w = long[long["dog_id"].isin(top["dog_id"])].pivot(index="dog_id", columns="surface",
                                                       values=["opportunities", "score", "score_per_100"])
    w.columns = [f"{s.lower()}_{m}" for m, s in w.columns]
    w = w.fillna({c: 0 for c in w.columns if c.endswith(("_opportunities", "_score"))})
    w = top.set_index("dog_id")[["player", "flag", "opportunities", "gk_score", "gk_score_unflagged_only"]].join(w)
    for s in SURFACES:
        k = s.lower()
        w[f"{k}_share_of_matches"] = w[f"{k}_opportunities"] / w["opportunities"]
    return w


if __name__ == "__main__":
    pm = pd.read_csv(config.PROCESSED_DIR / "gk_match_predictions.csv", parse_dates=["tourney_date"])
    overall = pd.read_csv(config.PROCESSED_DIR / "graph4_chart_data.csv")
    overall["flag"] = overall["flag"].fillna("none")   # guard in case "none" is read as missing
    long = surface_scores(pm)
    sanity_checks(pm, long, overall)

    w = wide_table(long, overall, TOP_N)
    print(f"\n--- Top {TOP_N} Giant Killers: score by surface (scores add up to gk_score) ---")
    cols = ["player", "flag", "gk_score"] + [f"{s.lower()}_score" for s in SURFACES]
    print(w[cols].round(1).to_string(index=False))
    print("\nShare of each player's underdog matches on each surface (compare with where the score comes from):")
    print(w[["player"] + [f"{s.lower()}_share_of_matches" for s in SURFACES]].round(2).to_string(index=False))
    print("\nScore per 100 underdog matches on each surface (NaN = never played there; small samples are noisy):")
    print(w[["player"] + [f"{s.lower()}_{m}" for s in SURFACES for m in ("opportunities", "score_per_100")]]
          .round(1).to_string(index=False))
    print("\nRanking changes: overall score vs score using only matches with neither comeback nor riser flag")
    print(w[["player", "flag", "gk_score", "gk_score_unflagged_only"]].round(1).to_string(index=False))

    print(f"\n--- Best Giant Killers on each surface (all players, min {MIN_SURFACE_OPPS} underdog matches there) ---")
    names = pm.drop_duplicates("dog_id").set_index("dog_id")["dog_name"]
    for s in SURFACES:
        b = long[(long["surface"] == s) & (long["opportunities"] >= MIN_SURFACE_OPPS)].copy()
        b["player"] = b["dog_id"].map(names)
        b["z"] = b["score"] / b["se"]
        print(f"\n{s} ({len(b)} players qualify):")
        print(b.sort_values("score", ascending=False).head(8)[["player", "opportunities", "upset_wins",
                                                               "expected_wins", "score", "z"]].round(2).to_string(index=False))

    # ---------------- figure: diverging stacked bars ----------------
    # Positive surface scores stack right of 0, negative ones stack left; the black tick is the net score.
    t = w.iloc[::-1]   # largest overall score at the top
    fig, ax = plt.subplots(figsize=(10.5, 9.6))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.83, bottom=0.12)
    ys = np.arange(len(t))
    pos_left, neg_left = np.zeros(len(t)), np.zeros(len(t))
    for s in SURFACES:
        v = t[f"{s.lower()}_score"].to_numpy()
        left = np.where(v >= 0, pos_left, neg_left)
        ax.barh(ys, v, left=left, height=0.62, color=SURFACE_COLORS[s], edgecolor=SURFACE_BG, linewidth=1.5)
        pos_left += np.clip(v, 0, None)
        neg_left += np.clip(v, None, 0)
    net = t["gk_score"].to_numpy()
    ax.scatter(net, ys, marker="|", s=260, color=INK, linewidths=2, zorder=3)
    for yv, right_end, n in zip(ys, pos_left, net):
        ax.annotate(f"{n:+.1f}", xy=(right_end, yv), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9.5, fontweight="bold", color=INK)
    for yv, (_, r) in zip(ys, t.iterrows()):
        ax.text(-0.015, yv + 0.13, r["player"], transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, color=INK)
        mix = "  ".join(f"{s[0]} {int(r[f'{s.lower()}_opportunities'])}" for s in SURFACES)
        tag = "" if r["flag"] == "none" else f"{r['flag'].capitalize()}  ·  "
        ax.text(-0.015, yv - 0.25, f"{tag}matches: {mix}", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=7.6, color=INK_2 if tag else MUTED)

    ax.set_yticks([])
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlim(min(0, neg_left.min()) - 1.5, pos_left.max() + 4)
    ax.set_ylim(-0.7, len(t) - 0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    ax.set_xlabel("Upset wins above expected, split by surface (segments add up to the net score)",
                  fontsize=9.5, color=INK_2)

    fig.text(0.02, 0.955, "Most Giant Killers beat their ranking on every surface; a few win their extra upsets on clay",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, f"Top {TOP_N} players from Graph 4, ATP main tour 2016-2025. Each segment is the "
             "surplus (or shortfall) of upset wins on one surface;", fontsize=10, color=INK_2)
    fig.text(0.02, 0.903, "the black tick marks the net Giant Killer Score. H / C / G = underdog matches on "
             "hard / clay / grass.", fontsize=10, color=INK_2)
    handles = [Patch(color=SURFACE_COLORS[s], label=s) for s in SURFACES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.89), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1.2, columnspacing=1.6)
    fig.text(0.02, 0.025, "Expected wins on each surface come from the same season-adjusted ranking + surface "
             "baseline as Graph 4, so the general surface effect is already removed.\nA large segment can "
             "reflect many matches on that surface, not only surface skill. Data: Jeff Sackmann / Tennis Abstract.",
             fontsize=7.6, color=MUTED, va="bottom")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "graph4b_surface_split.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    long.assign(player=long["dog_id"].map(names)).to_csv(config.PROCESSED_DIR / "graph4b_surface_data.csv", index=False)
    print(f"\nSaved -> {path} (+ .svg)\nSaved -> data/processed/graph4b_surface_data.csv")
