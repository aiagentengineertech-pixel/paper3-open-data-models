"""Paper 3 core protocol: identical tests on L-Town, Kitchener KIT 6 and Lower Hutt Konini_Parkway.
Demand level per network: the level at which 4 h/day supply short-changes customers (L-Town x1.5, Lower Hutt x1.0);
KIT 6 x1.0 (no shortfall possible, see notes). Supply model as Test 3 (PDA, Preq 20 m, 15-min steps, day 2).
A: guess one input at a time on the true layout (C=100 everywhere; pipes <200 mm set to 100 mm; +/-2.5 m iid ground-level
   noise; demand spread evenly; all combined) at 24 h and 4 h supply.
B: public-data rebuild (road-class and design-rule sizes) at 24 h and 4 h (KIT 6, Lower Hutt; L-Town has no real coordinates).
C: loggers read on a NORMAL 24 h day (head; logger ground level surveyed; 0.5 m noise) -> fit one roughness (K=1,3,5,8; 20 layouts)
   or one per district (3 location districts, K=8, 5 layouts, >=1 logger per district) -> predict 4 h supply.
Metrics: head RMSE, pressure RMSE, bias, % water delivered, junctions <90% (found / false alarms)."""
import os, sys, json, pickle, numpy as np, pandas as pd, wntr
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2
import tempfile
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, H + '/paper3_sensitivity')
import run_sensitivity as rs, run_iws as ri
OUT = H + '/core_protocol/results'; os.makedirs(OUT, exist_ok=True)
NET = {
 'LTOWN': dict(true=lambda: rs.load(), f=1.5, rebuilds={}),
 'KIT6': dict(true=lambda: wntr.network.WaterNetworkModel(H + '/kitchener/models/kit6_true.inp'), f=1.0,
              rebuilds={'osm_class': H + '/kitchener/models/kit6_osm_class.inp', 'osm_design': H + '/kitchener/models/kit6_osm_design.inp'}),
 'CHCH': dict(true=lambda: wntr.network.WaterNetworkModel(H + '/christchurch/models/Mt_Pleasant_3_true.inp'), f=1.0,
              rebuilds={'osm_class': H + '/christchurch/models/Mt_Pleasant_3_osm_class.inp', 'osm_design': H + '/christchurch/models/Mt_Pleasant_3_osm_design.inp'}),
 'LHUTT': dict(true=lambda: wntr.network.WaterNetworkModel(H + '/lowerhutt/models/Konini_Parkway_true.inp'), f=1.0,
              rebuilds={'osm_class': H + '/lowerhutt/models/Konini_Parkway_osm_class.inp', 'osm_design': H + '/lowerhutt/models/Konini_Parkway_osm_design.inp'})}
# Lower Hutt models are not distributed (Tiaki Wai Limited data, used with permission). Build them with lowerhutt/*.py
# from data obtained from Tiaki Wai Limited; until then the network is skipped.
if not os.path.exists(H + '/lowerhutt/models/Konini_Parkway_true.inp'): NET.pop('LHUTT')
def d_rough(wn):
    for p in wn.pipe_name_list: wn.get_link(p).roughness = 100.0
def d_diam(wn):
    for p in wn.pipe_name_list:
        l = wn.get_link(p)
        if l.diameter*1000 < 199: l.diameter = 0.1
def d_elev(wn):
    rng = np.random.default_rng(7)
    for j in wn.junction_name_list: wn.get_node(j).elevation += rng.normal(0, 2.5)
def d_even(wn):
    js = wn.junction_name_list; tot = sum(wn.get_node(j).demand_timeseries_list[0].base_value for j in js)
    for j in js: wn.get_node(j).demand_timeseries_list[0].base_value = tot/len(js)
DAM = {'rough_C100': [d_rough], 'diam_lt200_to_100': [d_diam], 'elev_noise_2.5m': [d_elev], 'demand_even': [d_even],
       'all_guessed': [d_diam, d_rough, d_even, d_elev]}
def model(net, kind, N, C=None, Cmap=None):
    """kind: 'true', a DAM key (true layout, damaged), or a rebuild key."""
    cfg = NET[net]
    wn = wntr.network.WaterNetworkModel(cfg['rebuilds'][kind]) if kind in cfg['rebuilds'] else cfg['true']()
    ri.scale_demand(wn, cfg['f']); wn = ri.to_iws(wn, N)
    for fn in DAM.get(kind, []): fn(wn)
    if C is not None:
        for p in wn.pipe_name_list: wn.get_link(p).roughness = float(C)
    if Cmap is not None:
        for p, c in Cmap.items(): wn.get_link(p).roughness = float(c)
    return wn
def sim(wn, tag):
    pre = os.path.join(tempfile.gettempdir(), f'core_{os.getpid()}_' + tag)
    r = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=pre); js = wn.junction_name_list; idx = r.node['pressure'].index
    for ext in ('.inp', '.bin', '.rpt', '.hyd'):
        try: os.remove(pre + ext)
        except OSError: pass
    req = pd.DataFrame({j: [wn.get_node(j).demand_timeseries_list[0].at(t) for t in idx] for j in js}, index=idx)
    return dict(p=r.node['pressure'][js], h=r.node['head'][js], q=r.node['demand'][js], req=req,
                xy=np.array([wn.get_node(j).coordinates for j in js]), names=js)
def win(df, N): return ri.window(df, N, 1)
def metrics(T, M, N, k=None):
    k = np.arange(len(T['names'])) if k is None else k
    pT, pM = win(T['p'], N).values, win(M['p'], N).values[:, k]; hT, hM = win(T['h'], N).values, win(M['h'], N).values[:, k]
    qT, rT = win(T['q'], N).sum().values, win(T['req'], N).sum().values; qM, rM = win(M['q'], N).sum().values, win(M['req'], N).sum().values
    with np.errstate(invalid='ignore', divide='ignore'): dT = qT/rT; dM = (qM/rM)[k]
    ok = np.isfinite(dT) & np.isfinite(dM); lT, lM = (dT < 0.9) & ok, (dM < 0.9) & ok
    return dict(head_rmse=float(np.sqrt(((hM-hT)**2).mean())), p_rmse=float(np.sqrt(((pM-pT)**2).mean())), p_bias=float((pM-pT).mean()),
                deliv_true=float(100*qT.sum()/rT.sum()), deliv_model=float(100*qM.sum()/rM.sum()),
                short_true=int(lT.sum()), short_found=int((lT & lM).sum()), false_alarms=int((lM & ~lT).sum()))
_truth = {}
def truth(net, N):
    fp = f'{OUT}/_truth_{net}_N{N}.pkl'
    if (net, N) not in _truth:
        if os.path.exists(fp): _truth[(net, N)] = pickle.load(open(fp, 'rb'))
        else: _truth[(net, N)] = sim(model(net, 'true', N), f'T{net}{N}'); pickle.dump(_truth[(net, N)], open(fp, 'wb'))
    return _truth[(net, N)]
def match(T, M): return cKDTree(M['xy']).query(T['xy'])[1]
if __name__ == '__main__':
    stage = sys.argv[1]
    if stage in ('A', 'B'):
        rows = []
        for net in [n for n in NET if n in os.environ.get('NETS', ','.join(NET)).split(',')]:
            kinds = list(DAM) if stage == 'A' else list(NET[net]['rebuilds'])
            for kind in kinds:
                for N in [24, 4]:
                    T = truth(net, N); M = sim(model(net, kind, N), f'{stage}{net}{kind}{N}')
                    r = dict(network=net, stage=stage, model=kind, supply_h=N, **metrics(T, M, N, match(T, M) if stage == 'B' else None)); rows.append(r)
                    print({a: (round(b, 2) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/stage{stage}' + ('_' + os.environ['NETS'] if 'NETS' in os.environ else '') + '.csv', index=False)
