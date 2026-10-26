"""Paper 3, Test 4b: tune roughness zone by zone (and by pipe group) with the same virtual loggers.
Truth and starting model as Test 4 (L-Town, demand x1.5, 4 h/day; C1 = all inputs guessed).
Zones: L-Town's pressure zones (split at PRVs and pump) with the large main zone split into 3 districts
by location (k-means on coordinates) -> 5 zones. Pipe groups: 'guessed' (pipes <200 mm, set to 100 mm in C1)
vs 'trunk' (kept). Calibration: start from the best single C, then coordinate descent over each zone's C
(grid 90-230, 2 sweeps); zones with no logger keep the start value. Layouts: K = 5, 8, 12 loggers,
stratified random (one per zone first, rest random), 5 layouts each, same meter noise as Test 4."""
import wntr, numpy as np, pandas as pd, json, os, sys, pickle, networkx as nx
from scipy.cluster.vq import kmeans2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sensitivity as rs, run_iws as ri, run_loggers as rl
OUT = os.path.join(rl.OUT, 'zonecal'); os.makedirs(OUT, exist_ok=True)
GRID = [90, 110, 130, 150, 170, 190, 210, 230]
def zones():
    wn = rs.load(); G = nx.Graph(wn.to_graph().to_undirected())
    G.remove_edges_from([(l.start_node_name, l.end_node_name) for _, l in wn.links() if l.link_type in ('Valve', 'Pump')])
    comps = sorted([c for c in nx.connected_components(G) if len(c) > 5], key=len, reverse=True)
    js = wn.junction_name_list; z = {}
    main = [j for j in js if j in comps[0]]; xy = np.array([wn.get_node(j).coordinates for j in main])
    _, lab = kmeans2(xy, 3, seed=3, minit='++')
    for j, l in zip(main, lab): z[j] = int(l)
    for k, c in enumerate(comps[1:]):
        for j in c:
            if j in js: z[j] = 3 + k
    for j in js: z.setdefault(j, 0)
    pz = {}
    for p in wn.pipe_name_list:
        l = wn.get_link(p); a = z.get(l.start_node_name, z.get(l.end_node_name, 0)); pz[p] = a
    return z, pz
def group_map():
    wn = rs.load(); return {p: (1 if wn.get_link(p).diameter*1000 < 199 else 0) for p in wn.pipe_name_list}
def run_with(Cmap, pmap, tag):
    wn = rs.load()
    for f in rs.SCEN['C1_combined_limited']: f(wn)
    for p in wn.pipe_name_list: wn.get_link(p).roughness = Cmap[pmap[p]]
    rl.ri.scale_demand(wn, rl.F); return rl.summarise(rl.ri.to_iws(wn, rl.N), tag)
def calibrate(idx, obs, pmap, ngroups, C0, tag):
    C = {g: C0 for g in range(ngroups)}; cache = {}
    def mis(Cd):
        key = tuple(Cd[g] for g in range(ngroups))
        if key not in cache: cache[key] = run_with(Cd, pmap, tag)
        M = cache[key]; return np.sqrt(((M['p'][:, idx] - obs)**2).mean()), M
    best, M = mis(C)
    for sweep in range(2):
        for g in range(ngroups):
            for v in GRID:
                trial = dict(C); trial[g] = v; e, Mt = mis(trial)
                if e < best - 1e-6: best, C, M = e, trial, Mt
    return C, M, len(cache)
def score(M, T):
    lowT = T['deliv'] < 0.9; ok = ~np.isnan(M['deliv']) & ~np.isnan(T['deliv']); lowM = M['deliv'] < 0.9
    return dict(rmse_p=float(np.sqrt(((M['p']-T['p'])**2).mean())), flagged=int(lowM[ok].sum()),
        false_alarms=int((lowM & ~lowT)[ok].sum()), true_found=int((lowM & lowT)[ok].sum()), agree_pct=float(100*(lowM == lowT)[ok].mean()))
def layouts(T, z):
    rng = np.random.default_rng(21); names = T['names']; zid = np.array([z[n] for n in names]); L = []
    for K in [5, 8, 12]:
        for r in range(5):
            idx = [int(rng.choice(np.where(zid == g)[0])) for g in range(5)]
            rest = [i for i in rng.permutation(len(names)) if i not in idx][:K-5]; L.append((K, r, idx + [int(i) for i in rest]))
    return L
if __name__ == '__main__':
    T = pickle.load(open(os.path.join(rl.OUT, '_truth.pkl'), 'rb')); z, pz = zones(); gm = group_map()
    if sys.argv[1] == 'info':
        import collections; print('junctions per zone', sorted(collections.Counter(z.values()).items()))
        print('pipes per group (1=guessed 100mm)', collections.Counter(gm.values())); sys.exit()
    L = layouts(T, z); B = rl.bank('C1_combined_limited'); Cs = [c for c in B if c != 'uncal']
    for job in map(int, sys.argv[1:]):
        fp = os.path.join(OUT, f'job{job}.json')
        if os.path.exists(fp): continue
        K, r, idx = L[job]; rng = np.random.default_rng(100 + job)
        obs = T['p'][:, idx] + rng.normal(0, 0.5, T['p'][:, idx].shape)
        mis = {c: np.sqrt(((B[c]['p'][:, idx] - obs)**2).mean()) for c in Cs}; c0 = min(mis, key=mis.get)
        res = dict(K=K, layout=r, loggers=[T['names'][i] for i in idx], C_global=c0, **{f'global_{k}': v for k, v in score(B[c0], T).items()})
        Cz, Mz, nz = calibrate(idx, obs, pz, 5, c0, f'zc{job}'); res.update(C_zone=Cz, **{f'zone_{k}': v for k, v in score(Mz, T).items()})
        Cg, Mg, ng = calibrate(idx, obs, gm, 2, c0, f'gc{job}'); res.update(C_group=Cg, **{f'group_{k}': v for k, v in score(Mg, T).items()})
        json.dump(res, open(fp, 'w'), indent=1)
        print(job, K, 'global', c0, round(res['global_rmse_p'], 2), res['global_false_alarms'], '| zone', Cz, round(res['zone_rmse_p'], 2), res['zone_false_alarms'], res['zone_true_found'],
              '| group', Cg, round(res['group_rmse_p'], 2), res['group_false_alarms'], res['group_true_found'], f'runs {nz}+{ng}', flush=True)
