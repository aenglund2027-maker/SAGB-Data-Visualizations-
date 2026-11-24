from pathlib import Path
import argparse
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Defaults assume this file lives in <repository>/src and CSVs live in <repository>/data.
REPO_ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description='Build ATP upset analyses for Graphs 1 and 3')
parser.add_argument('--data-dir', type=Path, default=REPO_ROOT / 'data',
                    help='Directory containing atp_matches_2014.csv ... atp_matches_2025.csv')
parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'outputs')
args = parser.parse_args()
DATA_ROOT = args.data_dir.expanduser().resolve()
OUT = args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
missing = [str(DATA_ROOT / f'atp_matches_{y}.csv') for y in range(2014,2026)
           if not (DATA_ROOT / f'atp_matches_{y}.csv').is_file()]
if missing:
    parser.error('Missing input CSV files. First missing: ' + missing[0])
YEARS=range(2014,2026)
raw=pd.concat([pd.read_csv(DATA_ROOT/f'atp_matches_{y}.csv',low_memory=False).assign(season=y) for y in YEARS],ignore_index=True)
raw['event_date']=pd.to_datetime(raw['tourney_date'].astype('Int64').astype(str),format='%Y%m%d',errors='coerce')
score=raw['score'].fillna('').astype(str)
bad=score.str.contains(r'\b(?:RET|W/O|WO|DEF|ABN|ABD|SUSP|UNF|CANCEL|INCOMPLETE)\b',case=False,regex=True)
# Eligible ATP tour level & grandslam only; no Davis Cup; remove unfinished and missing surface/dates
base=raw.loc[raw['tourney_level'].isin(['A','M','G','F','O']) & raw['surface'].isin(['Hard','Clay','Grass']) & raw['event_date'].notna() & ~bad & score.ne('') & raw['winner_id'].notna() & raw['loser_id'].notna()].copy()
base=base.sort_values(['event_date','tourney_id','match_num'],kind='stable').reset_index(drop=True)
print('Eligible completed matches with valid surfaces and player IDs (2014–2025):',len(base))
# Prior ten completed ATP-tour matches. All players on the same event-start date use history from earlier dates only.
history=defaultdict(lambda:deque(maxlen=10))
wn=[];ww=[];ln=[];lw=[]
for date,group in base.groupby('event_date',sort=True):
    updates=[]
    for i,row in group.iterrows():
        w=int(row['winner_id']); l=int(row['loser_id'])
        wh=history[w];lh=history[l]
        wn.append(len(wh)); ww.append(sum(wh)); ln.append(len(lh));lw.append(sum(lh))
        updates.append((w,l))
    for w,l in updates:
        history[w].append(1);history[l].append(0)
base['winner_prior_n']=wn;base['winner_prior_wins']=ww;base['loser_prior_n']=ln;base['loser_prior_wins']=lw
# Rankings stored in CSV are pre-match ATP rankings.
for c in ['winner_rank','loser_rank']:
    base[c]=pd.to_numeric(base[c],errors='coerce')
ranked=base.loc[(base['winner_rank']>0)&(base['loser_rank']>0)&(base['winner_rank']!=base['loser_rank'])].copy()
ranked['upset']=(ranked['winner_rank']>ranked['loser_rank']).astype(int)
ranked['favorite_rank']=ranked[['winner_rank','loser_rank']].min(axis=1)
ranked['underdog_rank']=ranked[['winner_rank','loser_rank']].max(axis=1)
ranked['rank_gap']=ranked['underdog_rank']-ranked['favorite_rank']
ranked['underdog_prior_n']=np.where(ranked['upset']==1,ranked['winner_prior_n'],ranked['loser_prior_n'])
ranked['underdog_prior_wins']=np.where(ranked['upset']==1,ranked['winner_prior_wins'],ranked['loser_prior_wins'])
ranked['favorite_prior_n']=np.where(ranked['upset']==0,ranked['winner_prior_n'],ranked['loser_prior_n'])
ranked['favorite_prior_wins']=np.where(ranked['upset']==0,ranked['winner_prior_wins'],ranked['loser_prior_wins'])
ranked['underdog_recent_win_pct']=np.where(ranked['underdog_prior_n']>0,ranked['underdog_prior_wins']/ranked['underdog_prior_n']*100,np.nan)
ranked['favorite_recent_win_pct']=np.where(ranked['favorite_prior_n']>0,ranked['favorite_prior_wins']/ranked['favorite_prior_n']*100,np.nan)
# same features available at match start, no outcome information.
def create_X(d):
  return np.column_stack([np.ones(len(d)), np.log(d['underdog_rank'].to_numpy()/d['favorite_rank'].to_numpy()), np.log(d['favorite_rank'].to_numpy()), np.log1p(d['rank_gap'].to_numpy()), (d['surface'].to_numpy()=='Clay').astype(float), (d['surface'].to_numpy()=='Grass').astype(float)])
train=ranked.loc[ranked['season']<=2022].copy()
test=ranked.loc[ranked['season'].between(2023,2025)].copy()
Xtr=create_X(train);Xte=create_X(test)
means=Xtr[:,1:4].mean(axis=0);std=Xtr[:,1:4].std(axis=0)
Xtr[:,1:4]=(Xtr[:,1:4]-means)/std
Xte[:,1:4]=(Xte[:,1:4]-means)/std
# numpy logistic regression with regularized IRLS, no sklearn dependency needed
def fit_logistic(X,y,lamb=1e-3,max_iter=100):
   beta=np.zeros(X.shape[1]); eye=np.eye(X.shape[1]);eye[0,0]=0
   for _ in range(max_iter):
       z=np.clip(X@beta,-35,35);p=1/(1+np.exp(-z)); w=np.maximum(p*(1-p),1e-7)
       h=(X.T*w)@X + lamb*eye
       grad=X.T@(y-p)-lamb*(eye@beta)
       delta=np.linalg.solve(h,grad)
       beta+=delta
       if np.max(np.abs(delta))<1e-9:break
   return beta
beta=fit_logistic(Xtr,train['upset'].to_numpy())
test['expected_upset_pct']=100/(1+np.exp(-np.clip(Xte@beta,-35,35)))
test['difference_pp']=test['upset']*100 - test['expected_upset_pct']
print('Train N',len(train),'Test N',len(test),'Test upset rate',100*test['upset'].mean())
# Graph 1 bins
bins=[0,10,25,50,100,200,np.inf]
labels=['1–9','10–24','25–49','50–99','100–199','200+']
test['rank_gap_bin']=pd.cut(test['rank_gap'],bins=bins,right=False,labels=labels)
g1=test.groupby(['surface','rank_gap_bin'],observed=True).agg(matches=('upset','size'),upsets=('upset','sum'),upset_rate=('upset','mean')).reset_index()
g1['upset_rate']=g1['upset_rate']*100
print('\nG1\n',g1.pivot(index='rank_gap_bin',columns='surface',values='upset_rate').round(1).to_string())
print('N\n',g1.pivot(index='rank_gap_bin',columns='surface',values='matches').to_string())
# Graph 3 bins, no min prior match requirement, missing as explicit group
formbins=[-0.0001,30,50,70,100.0001]
formlabels=['0–29%','30–49%','50–69%','70–100%']
# use right=False: 30% -> 30-49%; 50 ->50-69%; 70->70-100
form=pd.cut(test['underdog_recent_win_pct'],bins=formbins,right=False,labels=formlabels)
test['form_bin']=form.astype(object).fillna('No prior matches')
order=formlabels+['No prior matches']
g3=test.groupby('form_bin').agg(matches=('upset','size'),upsets=('upset','sum'),upset_rate=('upset','mean'),expected_rate=('expected_upset_pct','mean'),mean_underdog_rank=('underdog_rank','mean'),avg_prior_n=('underdog_prior_n','mean')).reindex(order).reset_index()
g3['upset_rate']*=100
g3['above_expectation_pp']=g3['upset_rate']-g3['expected_rate']
print('\nG3\n',g3.to_string(index=False,float_format=lambda x:f'{x:,.2f}'))
print('betas',beta)
# Save all numerical data needed to reproduce plots without raw match data.
g1.to_csv(OUT/'graph1_surface_ranking_gap_summary.csv',index=False)
g3.to_csv(OUT/'graph3_recent_form_summary.csv',index=False)
# Keep test match-level data for reproducibility of figures and review.
cols=['season','tourney_id','tourney_name','event_date','surface','winner_id','winner_name','loser_id','loser_name','favorite_rank','underdog_rank','rank_gap','rank_gap_bin','upset','underdog_prior_n','underdog_prior_wins','underdog_recent_win_pct','favorite_prior_n','favorite_recent_win_pct','form_bin','expected_upset_pct','difference_pp']
test[cols].to_csv(OUT/'graph1_graph3_match_level_2023_2025.csv',index=False)
