"""Paper 3, Test 4: how many pressure loggers fix the roughness error under intermittent supply?
Truth: L-Town, demand x1.5, 4 h/day supply (PDA), as Test 3 stressed case.
Starting models: R2 (only roughness guessed, C=100) and C1 (all inputs guessed).
Loggers read true pressure every 15 min in the supply window (day 2) + N(0, 0.5 m) meter noise.
Calibration: one global H-W C chosen from a grid (70-150, step 2.5) to minimise logger misfit.
Placement: random (20 draws) vs 'sensitive' (nodes where pressure changes most with C, >=400 m apart)
vs 'spread' (farthest-point spacing across the network, starting at the most sensitive node).
Evaluation on ALL junctions: pressure RMSE, junctions <90% delivery flagged vs true."""
import wntr, numpy as np, pandas as pd, json, os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sensitivity as rs, run_iws as ri
BASE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(BASE, 'results', 'loggers'); os.makedirs(OUT, exist_ok=True)
N = 4; F = 1.5; GRID = np.arange(70, 150.1, 2.5)
def build(name, C=None):
    wn = rs.load()
    for f in rs.SCEN[name]: f(wn)
    if C is not None: rs.rough_default(wn, C)
    ri.scale_demand(wn, F); return ri.to_iws(wn, N)
def summarise(wn, tag):
    r = ri.sim(wn, tag); p = ri.window(r['p'], N, 1); q = ri.window(r['q'], N, 1).sum(); rq = ri.window(r['req'], N, 1).sum()
    return dict(p=p.values.astype(np.float32), deliv=(q/rq.replace(0, np.nan)).values)
def bank(name):
    fp = os.path.join(OUT, f'_bank_{name}.pkl')
    if os.path.exists(fp): return pickle.load(open(fp, 'rb'))
    b = {float(C): summarise(build(name, C), f'lg{name}{C}') for C in GRID}
    b['uncal'] = summarise(build(name), f'lg{name}u'); pickle.dump(b, open(fp, 'wb')); return b
if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'truth':
        wn = rs.load(); ri.scale_demand(wn, F); wn = ri.to_iws(wn, N); T = summarise(wn, 'lgT')
        T['xy'] = np.array([wn.get_node(j).coordinates for j in wn.junction_name_list]); T['names'] = wn.junction_name_list
        pickle.dump(T, open(os.path.join(OUT, '_truth.pkl'), 'wb')); print('truth ok', T['p'].shape)
    elif what == 'bank': bank(sys.argv[2]); print('bank', sys.argv[2], 'ok')
    elif what == 'eval':
        T = pickle.load(open(os.path.join(OUT, '_truth.pkl'), 'rb')); rng = np.random.default_rng(11)
        lowT = T['deliv'] < 0.9; rows = []
        for name in ['R2_rough_all_C100', 'C1_combined_limited']:
            B = bank(name); Cs = [c for c in B if c != 'uncal']
            sens = np.abs(B[100.0]['p'] - B[140.0]['p']).mean(axis=0)
            order = np.argsort(-sens); smart = []
            for i in order:
                if all(np.hypot(*(T['xy'][i]-T['xy'][j])) >= 400 for j in smart): smart.append(i)
                if len(smart) == 10: break
            spread = [int(order[0])]
            while len(spread) < 10:
                d = np.min([np.hypot(*(T['xy'] - T['xy'][j]).T) for j in spread], axis=0); spread.append(int(np.argmax(d)))
            def score(M):
                ok = ~np.isnan(M['deliv']) & ~np.isnan(T['deliv']); lowM = M['deliv'] < 0.9
                return dict(rmse_p=float(np.sqrt(((M['p']-T['p'])**2).mean())), flagged=int(lowM[ok].sum()),
                            true_found=int((lowM & lowT)[ok].sum()), false_alarms=int((lowM & ~lowT)[ok].sum()),
                            agree_pct=float(100*(lowM == lowT)[ok].mean()))
            rows.append(dict(model=name, K=0, placement='none', C_cal=np.nan, **score(B['uncal'])))
            for K in [1, 2, 3, 5, 8]:
                sets = [('sensitive', smart[:K]), ('spread', spread[:K])] + [('random', list(rng.choice(len(T['names']), K, replace=False))) for _ in range(20)]
                for plc, idx in sets:
                    obs = T['p'][:, idx] + rng.normal(0, 0.5, T['p'][:, idx].shape)
                    mis = {c: np.sqrt(((B[c]['p'][:, idx] - obs)**2).mean()) for c in Cs}; c = min(mis, key=mis.get)
                    rows.append(dict(model=name, K=K, placement=plc, C_cal=c, loggers=';'.join(T['names'][i] for i in idx), **score(B[c])))
        df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'logger_calibration_all.csv'), index=False)
        g = df.groupby(['model', 'K', 'placement']).agg(C_cal=('C_cal', 'median'), rmse_p=('rmse_p', 'median'), rmse_p_p90=('rmse_p', lambda s: s.quantile(.9)),
              flagged=('flagged', 'median'), false_alarms=('false_alarms', 'median'), fa_p90=('false_alarms', lambda s: s.quantile(.9)),
              true_found=('true_found', 'median'), agree_pct=('agree_pct', 'median')).round(2)
        g.to_csv(os.path.join(OUT, 'logger_calibration_summary.csv')); print(g.to_string())
        print('true under-90%:', int(lowT.sum()), ' sensitive loggers C1:', [T['names'][i] for i in smart[:8]])
def eval_surveyed():
    """C1 model, but each logger site's ground level is surveyed: model pressure at the logger is
    corrected by the DEM elevation error at that node (head-based comparison)."""
    T = pickle.load(open(os.path.join(OUT, '_truth.pkl'), 'rb')); rng = np.random.default_rng(11); lowT = T['deliv'] < 0.9
    t = rs.load(); m = rs.load()
    for f in rs.SCEN['C1_combined_limited']: f(m)
    eerr = np.array([m.get_node(j).elevation - t.get_node(j).elevation for j in T['names']])
    B = bank('C1_combined_limited'); Cs = [c for c in B if c != 'uncal']
    sens = np.abs(B[100.0]['p'] - B[140.0]['p']).mean(axis=0); order = np.argsort(-sens); smart = []
    for i in order:
        if all(np.hypot(*(T['xy'][i]-T['xy'][j])) >= 400 for j in smart): smart.append(i)
        if len(smart) == 10: break
    spread = [int(order[0])]
    while len(spread) < 10:
        d = np.min([np.hypot(*(T['xy'] - T['xy'][j]).T) for j in spread], axis=0); spread.append(int(np.argmax(d)))
    rows = []
    for K in [1, 2, 3, 5, 8]:
        sets = [('sensitive', smart[:K]), ('spread', spread[:K])] + [('random', list(rng.choice(len(T['names']), K, replace=False))) for _ in range(20)]
        for plc, idx in sets:
            obs = T['p'][:, idx] + rng.normal(0, 0.5, T['p'][:, idx].shape)
            mis = {c: np.sqrt(((B[c]['p'][:, idx] + eerr[idx] - obs)**2).mean()) for c in Cs}; c = min(mis, key=mis.get)
            M = B[c]; ok = ~np.isnan(M['deliv']) & ~np.isnan(T['deliv']); lowM = M['deliv'] < 0.9
            rows.append(dict(model='C1 + logger sites surveyed', K=K, placement=plc, C_cal=c, rmse_p=float(np.sqrt(((M['p']-T['p'])**2).mean())),
                 flagged=int(lowM[ok].sum()), false_alarms=int((lowM & ~lowT)[ok].sum()), true_found=int((lowM & lowT)[ok].sum()), agree_pct=float(100*(lowM == lowT)[ok].mean())))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'logger_calibration_surveyed_all.csv'), index=False)
    g = df.groupby(['K', 'placement']).agg(C_cal=('C_cal', 'median'), rmse_p=('rmse_p', 'median'), flagged=('flagged', 'median'),
        false_alarms=('false_alarms', 'median'), fa_p90=('false_alarms', lambda s: s.quantile(.9)), true_found=('true_found', 'median'), agree_pct=('agree_pct', 'median')).round(2)
    g.to_csv(os.path.join(OUT, 'logger_calibration_surveyed_summary.csv')); print(g.to_string())
