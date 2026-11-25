"""SAGB figures 1 and 3 — full 2014–2025 retrospective ATP main-draw analysis.

Requirements: pandas, numpy. Raw yearly atp_matches_YYYY.csv files are NOT redistributed.
This script assumes data/ at repository root unless --data-dir is provided.

IMPORTANT: the baseline used in Graph 3 is a descriptive *in-sample* adjustment
for ranking and surface, NOT an out-of-sample forecast. The seven-variable PCA
analysis is not involved in these two graphs.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, deque
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description='Rebuild full-period SAGB upset analyses (2014–2025)')
parser.add_argument('--data-dir', type=Path, default=ROOT/'data', help='Directory with 2014–2025 yearly ATP match CSV files')
parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
args = parser.parse_args()
data_dir = args.data_dir.expanduser().resolve()
out_dir = args.output_dir.expanduser().resolve()
out_dir.mkdir(parents=True, exist_ok=True)
years = range(2014, 2026)
paths = [data_dir/f'atp_matches_{y}.csv' for y in years]
missing = [p.name for p in paths if not p.is_file()]
if missing:
    parser.error('Missing original match files: ' + ', '.join(missing))

matches = pd.concat(
    [pd.read_csv(p, low_memory=False).assign(season=y) for p,y in zip(paths,years)],
    ignore_index=True
)
matches['event_date'] = pd.to_datetime(matches['tourney_date'].astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
score = matches['score'].fillna('').astype(str)
unfinished = score.str.contains(r'\b(?:RET|W/O|WO|DEF|ABN|ABD|SUSP|UNF|CANCEL|INCOMPLETE)\b', case=False, regex=True)
# Regular tour-level main draws, Masters, Grand Slams, Tour Finals and Olympics.
valid = (matches['tourney_level'].isin(['A','M','G','F','O'])
         & matches['surface'].isin(['Hard','Clay','Grass'])
         & matches['event_date'].notna() & ~unfinished & score.ne('')
         & matches['winner_id'].notna() & matches['loser_id'].notna())
base = matches.loc[valid].sort_values(['event_date','tourney_id','match_num'], kind='stable').reset_index(drop=True)

# Pre-match form, no future leakage: earlier *tournament start dates* only. Since the
# annual ATP files contain tourney_date but no precise individual match date,
# matches with the SAME tourney_date cannot be reliably ordered relative to one another.
# Therefore all those matches receive profiles based on dates strictly before that date.
history = defaultdict(lambda: deque(maxlen=10))
win_n = np.empty(len(base), dtype=int); win_w = np.empty(len(base), dtype=int)
lose_n = np.empty(len(base), dtype=int); lose_w = np.empty(len(base), dtype=int)
for _, group in base.groupby('event_date', sort=True):
    updates = []
    for index, row in group.iterrows():
        winner = int(row['winner_id']); loser = int(row['loser_id'])
        past_w = history[winner]; past_l = history[loser]
        win_n[index] = len(past_w); win_w[index] = sum(past_w)
        lose_n[index] = len(past_l); lose_w[index] = sum(past_l)
        updates.append((winner,loser))
    for winner, loser in updates:
        history[winner].append(1); history[loser].append(0)
base['winner_prior_n'] = win_n
base['winner_prior_wins'] = win_w
base['loser_prior_n'] = lose_n
base['loser_prior_wins'] = lose_w

for c in ['winner_rank','loser_rank']:
    base[c] = pd.to_numeric(base[c], errors='coerce')
valid_ranks = ((base['winner_rank']>0) & (base['loser_rank']>0)
               & (base['winner_rank']!=base['loser_rank']))
df = base.loc[valid_ranks].copy()
df['upset'] = (df['winner_rank']>df['loser_rank']).astype(int)
df['favorite_rank'] = df[['winner_rank','loser_rank']].min(axis=1)
df['underdog_rank'] = df[['winner_rank','loser_rank']].max(axis=1)
df['rank_gap'] = df['underdog_rank']-df['favorite_rank']
df['underdog_prior_n'] = np.where(df['upset'].eq(1), df['winner_prior_n'], df['loser_prior_n'])
df['underdog_prior_wins'] = np.where(df['upset'].eq(1), df['winner_prior_wins'], df['loser_prior_wins'])
df['favorite_prior_n'] = np.where(df['upset'].eq(0), df['winner_prior_n'], df['loser_prior_n'])
df['favorite_prior_wins'] = np.where(df['upset'].eq(0), df['winner_prior_wins'], df['loser_prior_wins'])
df['underdog_recent_win_pct'] = np.where(df['underdog_prior_n']>0, 100*df['underdog_prior_wins']/df['underdog_prior_n'].replace(0,np.nan), np.nan)
df['favorite_recent_win_pct'] = np.where(df['favorite_prior_n']>0, 100*df['favorite_prior_wins']/df['favorite_prior_n'].replace(0,np.nan), np.nan)

# Graph 1: all 2014–2025 match records, not a 2023–2025 test-only subset.
gap_bins = [0,10,25,50,100,200,np.inf]
gap_labels = ['1–9','10–24','25–49','50–99','100–199','200+']
df['rank_gap_bin'] = pd.cut(df['rank_gap'], bins=gap_bins, right=False, labels=gap_labels)
g1 = df.groupby(['surface','rank_gap_bin'], observed=True).agg(
    matches=('upset','size'), upsets=('upset','sum'), upset_rate=('upset','mean')
).reset_index()
g1['upset_rate'] *= 100

# Graph 3: standardized historical form split. No minimum-match threshold.
# For a player with 1–9 earlier matches on strictly earlier event dates, use
# those recorded matches; if zero, place in 'No earlier recorded matches'.
form_bins = [-0.00001,30,50,70,100.0001]
form_labels = ['0–29%','30–49%','50–69%','70–100%']
df['form_bin'] = pd.cut(df['underdog_recent_win_pct'], bins=form_bins, right=False,
                        labels=form_labels).astype(object).fillna('No prior recorded matches')

# Retrospective ranking/surface benchmark estimated on all 2014–2025 outcomes.
# It is NOT out-of-sample; do not interpret as a backtested betting model.
def baseline_features(rows):
    return np.column_stack([
        np.ones(len(rows)),
        np.log(rows['underdog_rank'].to_numpy()/rows['favorite_rank'].to_numpy()),
        np.log(rows['favorite_rank'].to_numpy()),
        np.log1p(rows['rank_gap'].to_numpy()),
        rows['surface'].eq('Clay').to_numpy(dtype=float),
        rows['surface'].eq('Grass').to_numpy(dtype=float)
    ])
X = baseline_features(df)
means = X[:,1:4].mean(axis=0)
sds = X[:,1:4].std(axis=0)
assert np.all(sds>0)
X[:,1:4] = (X[:,1:4]-means)/sds

def logistic_fit(features, outcomes, penalty=1e-3, iters=100):
    beta=np.zeros(features.shape[1]); reg=np.eye(features.shape[1]);reg[0,0]=0
    for _ in range(iters):
        linear=np.clip(features@beta, -35,35)
        probs=1/(1+np.exp(-linear))
        weights=np.maximum(probs*(1-probs),1e-7)
        hess=(features.T*weights)@features + penalty*reg
        grad=features.T@(outcomes-probs) - penalty*(reg@beta)
        delta=np.linalg.solve(hess,grad)
        beta+=delta
        if np.max(np.abs(delta))<1e-9: break
    return beta

coef = logistic_fit(X, df['upset'].to_numpy())
df['expected_upset_pct'] = 100/(1+np.exp(-np.clip(X@coef,-35,35)))
df['difference_pp'] = df['upset']*100 - df['expected_upset_pct']
order = form_labels+['No prior recorded matches']
g3 = df.groupby('form_bin').agg(
    matches=('upset','size'), upsets=('upset','sum'),
    upset_rate=('upset','mean'),expected_rate=('expected_upset_pct','mean'),
    mean_underdog_rank=('underdog_rank','mean'),
    avg_prior_n=('underdog_prior_n','mean')
).reindex(order).reset_index()
g3['upset_rate'] *=100
g3['above_expectation_pp'] = g3['upset_rate']-g3['expected_rate']

# Output files intentionally use the SAME names as the two previously uploaded
# charts and summaries, so they replace the earlier 2023–2025 files on GitHub.
g1.to_csv(out_dir/'graph1_surface_ranking_gap_summary.csv', index=False)
g3.to_csv(out_dir/'graph3_recent_form_summary.csv', index=False)
cols=['season','tourney_id','tourney_name','event_date','surface','winner_id','winner_name',
      'loser_id','loser_name','favorite_rank','underdog_rank','rank_gap','rank_gap_bin',
      'upset','underdog_prior_n','underdog_prior_wins','underdog_recent_win_pct',
      'favorite_prior_n','favorite_recent_win_pct','form_bin','expected_upset_pct','difference_pp']
df[cols].to_csv(out_dir/'graph1_graph3_match_level_2014_2025.csv',index=False)

print(f'Raw match rows: {len(matches):,} | eligible completed rows: {len(base):,} | ranked matches: {len(df):,}')
print(f'Period: {df.season.min()}–{df.season.max()} | upset rate: {100*df.upset.mean():.2f}%')
print('Graph 1 summary:\n',g1.to_string(index=False))
print('Graph 3 summary:\n',g3.to_string(index=False))
print('Model: retrospective, descriptive rank+surface baseline (not prospective out-of-sample predictions).')
