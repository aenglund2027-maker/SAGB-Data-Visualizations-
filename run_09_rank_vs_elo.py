"""
Step 9 (new Graph 2): ATP ranking vs Elo disagreement.

Question: when the lower-ranked player is actually rated STRONGER by Elo, how often
does he win - and is that just because those matches have small ranking gaps?

  Elo agrees     : the higher-ranked player also has the higher pre-match Elo
  Elo disagrees  : the lower-ranked player has the higher pre-match Elo

Both groups are compared within ranking-gap bands (underdog rank / favorite rank),
so the comparison is not just "close matches vs lopsided matches".
Upset = lower-ranked player wins (same definition as everywhere else).
Pre-match Elo only uses earlier matches, so nothing here looks ahead.

Input:  data/processed/atp_model_dataset.csv
Output: outputs/figures/graph2_rank_vs_elo.png / .svg
        data/processed/graph2_chart_data.csv
Run from the project root:  python3 run_09_rank_vs_elo.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src import config

pd.set_option("display.width", 180)

# Ranking-gap bands, as underdog rank / favorite rank (e.g. 2.5 = ranked 2.5x lower)
RATIO_BINS = [1, 1.5, 2.5, 5, np.inf]
RATIO_LABELS = ["Close\n(under 1.5x)", "Moderate\n(1.5-2.5x)", "Large\n(2.5-5x)", "Huge\n(5x or more)"]
# Robustness: Elo is unreliable for players with few rated matches (everyone starts at 1500)
MIN_ELO_MATCHES = 20

# Style tokens (same as run_08_figures.py)
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
AGREE_COLOR, DISAGREE_COLOR = MUTED, "#eb6834"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "font.size": 10,
})


def wilson_ci(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n (well-behaved for small n)."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def load_sample():
    d = pd.read_csv(config.PROCESSED_DIR / "atp_model_dataset.csv", parse_dates=["tourney_date"], low_memory=False)
    print(f"Rows in model dataset: {len(d):,}")
    d = d[d["completed"] & d["upset_label_ok"] & (d["season"] >= config.MODEL_FIRST_SEASON)]
    print(f"Completed, rank label defined, season >= {config.MODEL_FIRST_SEASON}: {len(d):,}")

    ties = (d["elo_diff"] == 0).sum()
    d = d[d["elo_diff"] != 0].copy()
    print(f"Dropped {ties} exact Elo ties -> {len(d):,} matches")

    d["elo_view"] = np.where(d["elo_diff"] > 0, "agrees", "disagrees")   # elo_diff = fav Elo - dog Elo
    d["rank_ratio"] = d["dog_rank_eff"] / d["fav_rank_eff"]
    d["gap_band"] = pd.cut(d["rank_ratio"], RATIO_BINS, labels=RATIO_LABELS, right=False)
    d["elo_reliable"] = (d["fav_elo_n"] >= MIN_ELO_MATCHES) & (d["dog_elo_n"] >= MIN_ELO_MATCHES)
    return d


def summarise(d, by):
    g = d.groupby(by, observed=True)
    out = g.agg(matches=("upset", "size"), upsets=("upset", "sum"),
                elo_pred_upset_rate=("elo_prob_dog", "mean")).reset_index()
    out["upset_rate"] = out["upsets"] / out["matches"]
    out["ci_low"], out["ci_high"] = wilson_ci(out["upsets"], out["matches"])
    return out


def sanity_checks(d):
    print("\n--- Sanity checks ---")
    print(f"Upset label matches ranks on every row: {(d['upset'] == (d['winner_rank'] > d['loser_rank'])).all()}")
    print(f"Every match has rank ratio >= 1: {(d['rank_ratio'] >= 1).all()}")
    print(f"Matches with no gap band (should be 0): {d['gap_band'].isna().sum()}")
    print(f"\nElo disagrees with the ranking in {(d['elo_view'] == 'disagrees').mean():.1%} of matches")
    print("\nShare of matches where Elo disagrees, by ranking-gap band (disagreement should shrink as gaps grow):")
    share = d.groupby("gap_band", observed=True)["elo_view"].apply(lambda s: (s == "disagrees").mean())
    print(share.round(3).to_string())
    print(f"\nBoth players have >= {MIN_ELO_MATCHES} prior Elo matches in {d['elo_reliable'].mean():.1%} of matches")


if __name__ == "__main__":
    d = load_sample()
    sanity_checks(d)

    print("\n--- Overall upset rate by Elo view ---")
    print(summarise(d, ["elo_view"]).round(3).to_string(index=False))

    table = summarise(d, ["gap_band", "elo_view"])
    print("\n--- Upset rate by ranking-gap band and Elo view (95% Wilson intervals) ---")
    print("elo_pred_upset_rate = average upset probability Elo itself gave (a calibration check)")
    print(table.round(3).to_string(index=False))

    print(f"\n--- Robustness: only matches where both players have >= {MIN_ELO_MATCHES} prior Elo matches ---")
    print(summarise(d[d["elo_reliable"]], ["gap_band", "elo_view"]).round(3).to_string(index=False))

    print("\n--- By surface (does the pattern hold everywhere?) ---")
    print(summarise(d, ["surface", "elo_view"]).round(3).to_string(index=False))

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.73, bottom=0.2)
    x = np.arange(len(RATIO_LABELS))
    width = 0.36
    for offset, view, color in ((-width / 2, "agrees", AGREE_COLOR), (width / 2, "disagrees", DISAGREE_COLOR)):
        t = table[table["elo_view"] == view].set_index("gap_band").reindex(RATIO_LABELS)
        rate = 100 * t["upset_rate"]
        ax.bar(x + offset, rate, width=width - 0.03, color=color, linewidth=0)
        ax.errorbar(x + offset, rate, yerr=[rate - 100 * t["ci_low"], 100 * t["ci_high"] - rate],
                    fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        for xi, r, hi, n in zip(x + offset, rate, 100 * t["ci_high"], t["matches"]):
            ax.annotate(f"{r:.0f}%", xy=(xi, hi), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold", color=INK)
            ax.annotate(f"n={int(n):,}", xy=(xi, 0), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7.5, color=SURFACE if r > 6 else INK_2)

    ax.axhline(50, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(-0.48, 50.8, "50% = coin flip", ha="left", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, RATIO_LABELS, fontsize=9.5)
    ax.set_ylim(0, max(60, 100 * table["ci_high"].max() + 8))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylabel("Lower-ranked player wins", fontsize=9.5)
    ax.set_xlabel("Ranking gap (how many times lower the underdog is ranked)", fontsize=9.5, color=INK_2)

    n_dis = (d["elo_view"] == "disagrees").mean()
    fig.text(0.02, 0.945, "When Elo disagrees with the ranking, the lower-ranked player wins far more often",
             fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.02, 0.865, f"Share of matches won by the lower-ranked player, ATP main tour "
             f"{int(d['season'].min())}-{int(d['season'].max())} ({len(d):,} matches),\nsplit by ranking gap. "
             f"Elo rates the lower-ranked player as stronger in {n_dis:.0%} of matches.",
             fontsize=10, color=INK_2)
    handles = [Patch(color=AGREE_COLOR, label="Elo agrees with the ranking"),
               Patch(color=DISAGREE_COLOR, label="Elo rates the lower-ranked player as stronger")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.84), ncol=2, frameon=False,
               fontsize=9.5, handlelength=1.2, columnspacing=1.6)
    fig.text(0.02, 0.025, "Elo = chronological pre-match rating built only from earlier matches "
             "(FiveThirtyEight method, 2014 as warm-up). Whiskers = 95% Wilson intervals for each upset rate.\n"
             "Upset = lower-ranked player wins; matches with a missing ranking are excluded. "
             "Data: Jeff Sackmann / Tennis Abstract.", fontsize=7.6, color=MUTED, va="bottom")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "graph2_rank_vs_elo.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    table.to_csv(config.PROCESSED_DIR / "graph2_chart_data.csv", index=False)
    print(f"\nSaved -> {path} (+ .svg)\nSaved -> data/processed/graph2_chart_data.csv")
