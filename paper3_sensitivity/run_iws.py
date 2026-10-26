"""Paper 3, Test 3: intermittent supply (IWS).
Supply ON for N h/day from 05:00; customers draw the full daily volume in that window
(demand multiplier 24/N), OFF otherwise. Pressure-dependent demand (EPANET 2.2 PDA):
Pmin 0 m, Preq 20 m, exponent 0.5. Approximation: EPANET cannot model pipe emptying/filling;
PDA with a supply window is the usual approximation [to cite].
L-Town: 48 h, 15-min steps, day 2 evaluated. Each damaged input from Test 1 vs the true model.
KIT 6: true network vs OSM design-sized rebuild (24 h + warm-up day).
Metrics (supply hours only): pressure RMSE; delivery ratio (delivered/required volume)
network total and per node; nodes under 90% delivery true vs model (agreement)."""
import wntr, numpy as np, pandas as pd, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sensitivity as rs
BASE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(BASE, '..', 'kitchener', 'models')
START = 5; PREQ = 20.0
def iws_pattern(N, step_h=0.25):
    n = int(24/step_h); m = np.zeros(n)
    for i in range(n):
        h = i*step_h
        if (h - START) % 24 < N: m[i] = 24.0/N
    return list(m)
def to_iws(wn, N, hours=48):
    """Collapse each junction's demands to its 24 h average, apply the supply-window pattern, set PDA."""
    pat = iws_pattern(N)
    if f'IWS{N}' not in wn.pattern_name_list: wn.add_pattern(f'IWS{N}', pat)
    for j in wn.junction_name_list:
        nd = wn.get_node(j); avg = 0.0
        for d in nd.demand_timeseries_list:
            p = d.pattern; avg += d.base_value * (np.mean(p.multipliers) if p is not None else 1.0)
        nd.demand_timeseries_list.clear(); nd.add_demand(avg, f'IWS{N}')
    o = wn.options
    o.time.duration = hours*3600; o.time.hydraulic_timestep = 900; o.time.pattern_timestep = 900
    o.time.report_timestep = 900; o.time.quality_timestep = 900
    o.hydraulic.demand_model = 'PDD'; o.hydraulic.minimum_pressure = 0.0
    o.hydraulic.required_pressure = PREQ; o.hydraulic.pressure_exponent = 0.5
    o.hydraulic.trials = 200; o.hydraulic.unbalanced = 'CONTINUE'; o.hydraulic.unbalanced_value = 20
    return wn
def sim(wn, tag):
    r = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=__import__('os').path.join(__import__('tempfile').gettempdir(), f'iws_{tag}'))
    js = wn.junction_name_list
    req = pd.DataFrame({j: [wn.get_node(j).demand_timeseries_list[0].at(t) for t in r.node['demand'].index] for j in js}, index=r.node['demand'].index)
    return dict(p=r.node['pressure'][js], q=r.node['demand'][js], req=req)
def window(df, N, day):
    t = df.index/3600.0; on = ((t - START) % 24 < N) & (t >= 24*day) & (t < 24*(day+1))
    return df[on]
def metrics(T, M, N, day, mapidx=None):
    pT, pM = window(T['p'], N, day), window(M['p'], N, day)
    qT, qM = window(T['q'], N, day).sum(), window(M['q'], N, day).sum()
    rT, rM = window(T['req'], N, day).sum(), window(M['req'], N, day).sum()
    if mapidx is not None:
        pM = pM.iloc[:, mapidx]; pM.columns = pT.columns
        dM = (qM/rM.replace(0, np.nan)).iloc[mapidx].values
    else: dM = (qM/rM.replace(0, np.nan)).values
    dT = (qT/rT.replace(0, np.nan)).values; ok = ~np.isnan(dT) & ~np.isnan(dM)
    err = (pM.values - pT.values)
    lowT, lowM = dT[ok] < 0.9, dM[ok] < 0.9
    return dict(N=N, rmse_p_m=float(np.sqrt((err**2).mean())), bias_p_m=float(err.mean()),
        min_p_true=float(pT.values.min()), min_p_model=float(pM.values.min()),
        deliv_true_pct=float(100*qT.sum()/rT.sum()), deliv_model_pct=float(100*qM.sum()/rM.sum()),
        node_deliv_rmse_pct=float(100*np.sqrt(((dM[ok]-dT[ok])**2).mean())),
        under90_true=int(lowT.sum()), under90_model=int(lowM.sum()), under90_both=int((lowT & lowM).sum()),
        under90_agree_pct=float(100*(lowT == lowM).mean()))
DAM = ['R2_rough_all_C100', 'D1_diam_trunk_only_100', 'E2_elev_noise_2.5m', 'Q1_demand_even', 'C1_combined_limited']
def ltown(N, names):
    out = os.path.join(BASE, 'results', 'iws', f'ltown_N{N}.json')
    res = json.load(open(out)) if os.path.exists(out) else {}
    import pickle; tp = os.path.join(BASE, 'results', 'iws', f'_ltown_true_N{N}.pkl')
    if os.path.exists(tp): T = pickle.load(open(tp, 'rb'))
    else:
        wn = to_iws(rs.load(), N); wntr.network.write_inpfile(wn, os.path.join(BASE, 'models', 'iws', f'LTOWN_true_IWS{N}.inp'), units='CMH')
        T = sim(wn, f'lt{N}'); pickle.dump(T, open(tp, 'wb'))
    res['TRUE'] = metrics(T, T, N, 1)
    for nm in names:
        t = time.time(); wn = rs.load()
        for f in rs.SCEN[nm]: f(wn)
        M = sim(to_iws(wn, N), f'lt{N}{nm}'); res[nm] = metrics(T, M, N, 1); res[nm]['run_s'] = round(time.time()-t, 1)
        print(N, nm, {k: round(v, 2) for k, v in res[nm].items() if k != 'N'}, flush=True)
    json.dump(res, open(out, 'w'), indent=1); print(N, 'TRUE', {k: round(v, 2) for k, v in res['TRUE'].items()})
def kit(N, model='kit6_osm_design.inp'):
    from scipy.spatial import cKDTree
    T_wn = to_iws(wntr.network.WaterNetworkModel(os.path.join(KIT, 'kit6_true.inp')), N)
    M_wn = to_iws(wntr.network.WaterNetworkModel(os.path.join(KIT, model)), N)
    T, M = sim(T_wn, f'k{N}t'), sim(M_wn, f'k{N}m')
    xyT = np.array([T_wn.get_node(j).coordinates for j in T_wn.junction_name_list])
    xyM = np.array([M_wn.get_node(j).coordinates for j in M_wn.junction_name_list])
    _, k = cKDTree(xyM).query(xyT)
    r = metrics(T, M, N, 1, mapidx=k); json.dump(r, open(os.path.join(BASE, 'results', 'iws', f'kit6_N{N}' + ('' if model=='kit6_osm_design.inp' else '_'+model[:-4]) + '.json'), 'w'), indent=1)
    print('KIT6', N, model, {a: round(b, 2) for a, b in r.items()})
if __name__ == '__main__':
    what, N = sys.argv[1], int(sys.argv[2])
    if what == 'ltown': ltown(N, sys.argv[3:] or DAM)
    else: kit(N, *sys.argv[3:])
def scale_demand(wn, f):
    for j in wn.junction_name_list:
        for d in wn.get_node(j).demand_timeseries_list: d.base_value *= f
def ltown_stressed(N, f=1.5):
    wn = rs.load(); scale_demand(wn, f); T = sim(to_iws(wn, N), f'lts{N}'); res = {'TRUE': metrics(T, T, N, 1)}
    for nm in DAM:
        wn = rs.load()
        for fn in rs.SCEN[nm]: fn(wn)
        scale_demand(wn, f); res[nm] = metrics(T, sim(to_iws(wn, N), f'lts{N}{nm}'), N, 1)
    json.dump(res, open(os.path.join(BASE, 'results', 'iws', f'ltown_x{f}_N{N}.json'), 'w'), indent=1)
    for k, v in res.items(): print(f'x{f} N{N}', k, {a: round(b, 1) for a, b in v.items() if a in ('rmse_p_m','bias_p_m','min_p_model','deliv_model_pct','node_deliv_rmse_pct','under90_true','under90_model','under90_both','under90_agree_pct')})
