"""
Hook for merging my partner's shared PCA/cluster style labels (Graphs 1-2).
No clustering is done in this project; labels come only from the partner's output.

Expected file (CSV), one row per player per period the label applies from:
    player_id   - Sackmann player id (same as winner_id / loser_id)
    valid_from  - date the label becomes known (YYYY-MM-DD)
    style_label - the partner's cluster name / id

For prediction (Graph 3), a label may only be used if it was computable before
the match: we attach the most recent label with valid_from STRICTLY before the
tournament date (pandas merge_asof, allow_exact_matches=False).

If the partner provides one static label per player (computed from the whole
2014-2025 period), pass static=True. That is fine for descriptive heatmaps but
leaks future information into prediction, so a warning is printed.
"""
import pandas as pd


def load_style_labels(path, static=False):
    labels = pd.read_csv(path)
    required = {"player_id", "style_label"} | (set() if static else {"valid_from"})
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"style label file is missing columns: {sorted(missing)}")
    if not static:
        labels["valid_from"] = pd.to_datetime(labels["valid_from"])
    return labels


def merge_style_labels(matches, labels, sides=("fav", "dog"), static=False):
    """Add {side}_style for each side (e.g. fav_style, dog_style) using pre-match labels only."""
    out = matches.copy()
    if static:
        print("WARNING: static style labels use full-period data; do not use them as predictive features.")
        lab = labels.set_index("player_id")["style_label"]
        for side in sides:
            out[f"{side}_style"] = out[f"{side}_id"].map(lab)
        return out

    lab = labels.sort_values("valid_from")
    out["_row"] = range(len(out))
    for side in sides:
        left = out[["_row", f"{side}_id", "tourney_date"]].rename(columns={f"{side}_id": "player_id"})
        left = left.sort_values("tourney_date")
        merged = pd.merge_asof(left, lab[["player_id", "valid_from", "style_label"]],
                               left_on="tourney_date", right_on="valid_from", by="player_id",
                               allow_exact_matches=False, direction="backward")
        out[f"{side}_style"] = merged.set_index("_row")["style_label"].reindex(out["_row"]).to_numpy()
    print("Style label coverage:", {s: f"{out[f'{s}_style'].notna().mean():.1%}" for s in sides})
    return out.drop(columns="_row")
