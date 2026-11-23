"""
Step 8: publication figures for Graph 3 and Graph 4 (matplotlib, PNG + SVG).

Inputs:  data/processed/graph3_results.csv, graph3_bootstrap.csv     (run_07_matchup.py)
         data/processed/gk_match_predictions.csv, atp_model_dataset.csv, atp_matches_clean.csv
Outputs: outputs/figures/graph3_model_comparison.png / .svg
         outputs/figures/graph4_giant_killers.png / .svg
         data/processed/graph4_chart_data.csv   (exact numbers plotted, with flags)
Run from the project root:  python3 run_08_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from src import config
from src.flags import flag_opportunities, player_flags
from src.giant_killers import giant_killer_table

# ---------------------------------------------------------------------------
# Style tokens (reference data-viz palette, light mode)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (validated all-pairs, light)
BLUE_LIGHT = "#86b6ef"                        # blue ramp step 250 (Graph 3: "same as profiles, plus")

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "axes.edgecolor": AXIS, "font.size": 10,
})


def style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, labelsize=9)


# ---------------------------------------------------------------------------
# Graph 3: baseline vs. profiles vs. matchups
# ---------------------------------------------------------------------------
def graph3(path):
    res = pd.read_csv(config.PROCESSED_DIR / "graph3_results.csv", index_col=0)
    boot = pd.read_csv(config.PROCESSED_DIR / "graph3_bootstrap.csv")
    auc_ci = boot[boot["metric"] == "auc"].set_index(["strength", "comparison"])
    const = res.loc["constant (no information)"]

    groups = [("log_rank", "ATP ranking models"), ("elo", "Elo rating models")]
    # (stage, bootstrap comparison vs the row above, row label, bar color)
    stages = [("baseline", None, "Strength + surface (baseline)", MUTED),
              ("additive", "additive vs baseline", "+ player profiles", SERIES[0]),
              ("matchup", "matchup vs additive", "+ profiles + matchup interactions", BLUE_LIGHT)]

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    fig.subplots_adjust(left=0.27, right=0.60, top=0.79, bottom=0.255)
    row = blended_transform_factory(fig.transFigure, ax.transData)   # x in figure units, y in data rows
    col_gain, col_ll, col_brier = 0.635, 0.875, 0.965

    y, yticks, ylabels = 0.0, [], []
    for g, (strength, header) in enumerate(groups):
        ax.text(-0.02, y, header, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10.5, fontweight="bold", color=INK)
        if g == 0:
            for x, label, ha in ((col_gain, "Change in AUC vs row above", "left"),
                                 (col_ll, "Log loss", "right"), (col_brier, "Brier", "right")):
                ax.text(x, y, label, transform=row, ha=ha, va="center", fontsize=8.5, color=MUTED)
        y += 1
        for stage, comp, label, color in stages:
            r = res.loc[f"{strength}:{stage}"]
            ax.barh(y, r["auc"] - 0.5, left=0.5, height=0.64, color=color, linewidth=0)
            ax.annotate(f"{r['auc']:.3f}", xy=(r["auc"], y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=10, fontweight="bold", color=INK)
            if comp:
                ci = auc_ci.loc[(strength, comp)]
                clear = ci["ci_low"] > 0
                ax.text(col_gain, y, f"{ci['improvement']:+.3f}", transform=row, va="center", fontsize=9.5,
                        color=INK if clear else MUTED, fontweight="bold" if clear else "normal")
                ax.text(col_gain + 0.052, y, "clear gain" if clear else "no clear gain", transform=row,
                        va="center", fontsize=9.5, color=INK_2 if clear else MUTED)
            ax.text(col_ll, y, f"{r['log_loss']:.4f}", transform=row, ha="right", va="center", fontsize=9.5, color=INK_2)
            ax.text(col_brier, y, f"{r['brier']:.4f}", transform=row, ha="right", va="center", fontsize=9.5, color=INK_2)
            yticks.append(y)
            ylabels.append(label)
            y += 1
        y += 0.5
    ax.set_yticks(yticks, ylabels, fontsize=10)
    ax.set_ylim(y - 0.9, -0.6)
    ax.set_xlim(0.5, 0.68)
    ax.set_xticks(np.arange(0.50, 0.681, 0.03), ["0.50\nchance"] + [f"{v:.2f}" for v in np.arange(0.53, 0.681, 0.03)])
    style_axis(ax)
    ax.spines["bottom"].set_visible(False)
    ax.axvline(0.5, color=INK_2, linewidth=1)
    ax.set_xlabel("AUC (higher = better at telling upsets from non-upsets)", fontsize=9.5, color=INK_2)

    fig.text(0.02, 0.952, "Player profiles sharpen upset predictions; matchup terms add no clear extra gain",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.908, "How well each model separates upsets from non-upsets in ATP main-tour matches it never saw: "
             f"test set {config.TEST_SEASONS[0]}-{config.TEST_SEASONS[-1]},", fontsize=10.5, color=INK_2)
    fig.text(0.02, 0.877, f"{int(const['n']):,} matches, {const['upset_rate']:.0%} won by the lower-ranked player. "
             "Log loss and Brier score (lower = better) tell the same story.", fontsize=10.5, color=INK_2)
    handles = [Patch(color=MUTED, label="Strength + surface"), Patch(color=SERIES[0], label="+ player profiles"),
               Patch(color=BLUE_LIGHT, label="+ profiles + matchup interactions")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1.2, columnspacing=1.6)
    fig.text(0.02, 0.025,
             "AUC = probability the model rates a randomly chosen upset as likelier than a randomly chosen non-upset; "
             "0.50 is chance, 1.00 is perfect.\n"
             "Change in AUC: paired bootstrap on the test matches (2,000 resamples); \"clear gain\" = 95% interval "
             "entirely above zero. The interaction intervals cross zero for AUC,\nlog loss and Brier alike. "
             f"A model with no information scores AUC 0.500, log loss {const['log_loss']:.4f}, Brier {const['brier']:.4f}. "
             "Logistic regressions trained on 2015-22.\n"
             "Profiles = each player's prior 52-week ace, double-fault, first-serve-in, first- and second-serve-won, "
             "service-points-won and return-points-won rates.\n"
             "Interactions = underdog return vs favorite serve, underdog serve vs favorite return, underdog return vs "
             "favorite second serve. Data: Jeff Sackmann / Tennis Abstract.",
             fontsize=7.6, color=MUTED, va="bottom", linespacing=1.45)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graph 4: Giant Killers
# ---------------------------------------------------------------------------
FLAG_STYLE = {"none": (SERIES[0], None),
              "comeback": (SERIES[1], "comeback"),
              "riser": (SERIES[2], "riser")}


def giant_killer_data():
    clean = pd.read_csv(config.CLEAN_MATCHES_PATH, parse_dates=["tourney_date"], low_memory=False)
    pm = pd.read_csv(config.PROCESSED_DIR / "gk_match_predictions.csv", parse_dates=["tourney_date"])
    ages = pd.read_csv(config.PROCESSED_DIR / "atp_model_dataset.csv", usecols=["match_id", "dog_age"])
    flagged = flag_opportunities(pm.merge(ages, on="match_id", how="left"), clean)
    prob = f"p_{config.GK_BASELINE}"
    table = giant_killer_table(flagged, prob, config.GK_MIN_OPPORTUNITIES).join(player_flags(flagged))
    # Robustness: the same score using only matches with neither flag
    clean_opps = flagged[~flagged["comeback"] & ~flagged["riser"]]
    unflagged = giant_killer_table(clean_opps, prob, config.GK_MIN_OPPORTUNITIES)
    table = table.join(unflagged[["opportunities", "gk_score", "z"]], rsuffix="_unflagged_only")
    return flagged, table


def graph4(table, path, top_n=20):
    table = table.sort_values("gk_score", ascending=False)   # rank on all qualifiers, display top_n
    t = table.head(top_n).iloc[::-1]   # largest at top
    fig, ax = plt.subplots(figsize=(10.5, 10))
    fig.subplots_adjust(left=0.375, right=0.97, top=0.82, bottom=0.15)
    ys = np.arange(len(t))
    lows = t["gk_score"] - 1.96 * t["se"]
    highs = t["gk_score"] + 1.96 * t["se"]
    for yv, (_, r), lo, hi in zip(ys, t.iterrows(), lows, highs):
        color, tag = FLAG_STYLE[r["flag"]]
        ax.barh(yv, r["gk_score"], height=0.6, color=color, edgecolor=SURFACE, linewidth=2)
        # Approximate expected-variation range (score +/- 1.96 se): thin line with short end caps
        ax.plot([lo, hi], [yv, yv], color=INK_2, linewidth=1.1, solid_capstyle="butt", zorder=3)
        ax.plot([lo, lo], [yv - 0.12, yv + 0.12], color=INK_2, linewidth=1.1, zorder=3)
        ax.plot([hi, hi], [yv - 0.12, yv + 0.12], color=INK_2, linewidth=1.1, zorder=3)
        # Offsets in points (not data units) so labels never collide regardless of bar length
        ax.annotate(f"{r['gk_score']:+.1f}", xy=(hi, yv), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK, fontweight="bold")
        if lo <= 0:
            ax.annotate("range includes 0", xy=(hi, yv), xytext=(40, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MUTED)
        # Row label: name, then flag tag + counts
        ax.text(-0.015, yv + 0.12, r["player"], transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, color=INK)
        detail = (f"{int(r['upset_wins'])} upset wins vs {r['expected_wins']:.1f} expected"
                  f"  ·  {int(r['opportunities'])} opportunities")
        if tag:
            detail = f"{tag.capitalize()}  ·  " + detail
        ax.text(-0.015, yv - 0.24, detail, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.6, color=INK_2 if tag else MUTED)
    ax.set_yticks([])
    ax.axvline(0, color=INK_2, linewidth=1)
    # The axis must include the full luck intervals, including any that dip below zero
    ax.set_xlim(min(0, lows.min()) - 1, highs.max() + 4.5)
    ax.set_ylim(-0.7, len(t) - 0.3)
    style_axis(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    ax.text(0.3, len(t) - 0.45, "0 = exactly as many upsets as expected", fontsize=8, color=MUTED, va="bottom")
    ax.set_xlabel("Giant Killer Score = actual upset wins − expected upset wins", fontsize=9.5, color=INK_2)

    fig.text(0.02, 0.955, "Giant Killers: who beats higher-ranked players more than expected?",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, f"Top {top_n} by upset wins beyond what the ATP-ranking gap and surface predicted, "
             "ATP main tour 2016-2025. An opportunity is a match", fontsize=10, color=INK_2)
    fig.text(0.02, 0.903, "played as the lower-ranked player. Lines show an approximate expected-variation range "
             "(not a full 95% confidence interval).", fontsize=10, color=INK_2)
    handles = [Patch(color=SERIES[0], label="No flag"),
               Patch(color=SERIES[1], label="Comeback: ranked far below his recent peak"),
               Patch(color=SERIES[2], label="Riser: young, ranking soared the next year")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.885), ncol=3, frameon=False,
               fontsize=9, handlelength=1.2, columnspacing=1.2)
    fig.text(0.02, 0.025, "Expected wins = sum of season-adjusted baseline probabilities: a logistic model (log "
             "ranking gap + surface) predicts each season from earlier seasons only (2014-15 serve as\nhistory), "
             "then each season is re-centered so its expected upsets equal its actual upsets. Range = score ± 1.96 × "
             "match-to-match chance variation given those probabilities;\nit ignores uncertainty in the model "
             f"itself. Ranked among all {len(table)} players with {config.GK_MIN_OPPORTUNITIES}+ opportunities "
             "(the top 20 is identical with no minimum).\nComeback = best ranking in prior 3 years was top-30 and at least 3x better than current; "
             "riser = age 23 or younger and ranking a year later at least 2x better.\nA player is flagged when at "
             f"least {config.FLAG_PLAYER_SHARE:.0%} of his matches as underdog meet the definition. "
             "Data: Jeff Sackmann / Tennis Abstract.", fontsize=7.6, color=MUTED, va="bottom")
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))   # editable vector copy for the article layout
    plt.close(fig)


if __name__ == "__main__":
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    g3 = config.FIGURES_DIR / "graph3_model_comparison.png"
    graph3(g3)
    print(f"Saved -> {g3}")

    flagged, table = giant_killer_data()
    print(f"\nFlagged opportunities: comeback {flagged['comeback'].mean():.1%}, riser {flagged['riser'].mean():.1%} "
          f"(next-year ranking unavailable for {flagged['dog_rank_next_year'].isna().mean():.1%}, incl. all of 2025)")
    print("Player flags among qualifiers:", table["flag"].value_counts().to_dict())
    cols = ["player", "opportunities", "gk_score", "se", "z", "comeback_share", "riser_share", "flag",
            "gk_score_unflagged_only", "opportunities_unflagged_only"]
    print("\nTop 20 as plotted (plus score using only unflagged matches):")
    print(table.head(20)[cols].round(2).to_string())
    print("\nRobustness — top 10 using only matches with neither flag:")
    alt = table.dropna(subset=["gk_score_unflagged_only"]).sort_values("gk_score_unflagged_only", ascending=False)
    print(alt.head(10)[["player", "opportunities_unflagged_only", "gk_score_unflagged_only",
                        "z_unflagged_only"]].round(2).to_string())

    table.to_csv(config.PROCESSED_DIR / "graph4_chart_data.csv")
    g4 = config.FIGURES_DIR / "graph4_giant_killers.png"
    graph4(table, g4)
    print(f"\nSaved -> {g4}\nSaved -> data/processed/graph4_chart_data.csv")
