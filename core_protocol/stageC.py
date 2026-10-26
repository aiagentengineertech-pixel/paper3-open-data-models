"""Stage C of the core protocol: loggers on a normal day -> roughness fit -> predict 4 h supply. See core.py."""
import os, sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from core import *
GRID = list(range(60, 231, 5)); ZGRID = list(range(60, 221, 20)); NOISE = 0.5
START = {'LTOWN': ['all_guessed'], 'CHCH': ['all_guessed', 'osm_class', 'osm_design'], 'KIT6': ['all_guessed', 'osm_class', 'osm_design'], 'LHUTT': ['all_guessed', 'osm_class', 'osm_design']}
def districts(net):
    T = truth(net, 24); _, lab = kmeans2(T['xy'], 3, seed=3, minit='++')
    cent = np.array([T['xy'][lab == g].mean(axis=0) for g in range(3)]); return lab, cent
def pipe_zones(wn, cent):
    t = cKDTree(cent); out = {}
    for p in wn.pipe_name_list:
        l = wn.get_link(p); a = np.array(l.start_node.coordinates); b = np.array(l.end_node.coordinates)
        out[p] = int(t.query((a+b)/2)[1])
    return out
def bank(net, kind):
    fp = f'{OUT}/_bank_{net}_{kind}.pkl'
    if os.path.exists(fp): return pickle.load(open(fp, 'rb'))
    B = {c: win(sim(model(net, kind, 24, C=c), f'b{net}{kind}{c}')['h'], 24).values.astype(np.float32) for c in GRID}
    pickle.dump(B, open(fp, 'wb')); return B
def layouts(net, K, n, strat, seed):
    T = truth(net, 24); rng = np.random.default_rng(seed); L = []; lab, _ = districts(net)
    for _ in range(n):
        idx = [int(rng.choice(np.where(lab == g)[0])) for g in range(3)] if strat else []
        rest = [int(i) for i in rng.permutation(len(T['names'])) if i not in idx][:K-len(idx)]; L.append(idx + rest)
    return L
def obs_for(net, idx, seed):
    T = truth(net, 24); rng = np.random.default_rng(seed); h = win(T['h'], 24).values[:, idx]
    return h + rng.normal(0, NOISE, h.shape)
if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'bank':
        net, kind = sys.argv[2], sys.argv[3]; bank(net, kind); print('bank', net, kind, 'ok', flush=True)
    elif stage == 'global':
        net = sys.argv[2]; rows = []; T4 = truth(net, 4); T24 = truth(net, 24); pred = {}
        for kind in START[net]:
            B = bank(net, kind); M0 = sim(model(net, kind, 24, C=GRID[0]), 'm0'); k = match(T24, M0)
            base4 = sim(model(net, kind, 4), f'g{net}{kind}base'); r0 = metrics(T4, base4, 4, k)
            rows.append(dict(network=net, start=kind, method='none', K=0, layout=0, C=None, **r0))
            for K in [1, 3, 5, 8]:
                for li, idx in enumerate(layouts(net, K, 20, False, 100 + K)):
                    ob = obs_for(net, idx, 1000 + 37*K + li)
                    c = min(GRID, key=lambda c: np.sqrt(((B[c][:, k[idx]] - ob)**2).mean()))
                    if (kind, c) not in pred: pred[(kind, c)] = metrics(T4, sim(model(net, kind, 4, C=c), f'g{net}{kind}{c}'), 4, k)
                    rows.append(dict(network=net, start=kind, method='global', K=K, layout=li, C=c, **pred[(kind, c)]))
            print(net, kind, 'global done', flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/stageC_global_{net}.csv', index=False)
    elif stage == 'zone':
        net, kind = sys.argv[2], sys.argv[3]; jobs = list(map(int, sys.argv[4:]))
        T4 = truth(net, 4); T24 = truth(net, 24); B = bank(net, kind); _, cent = districts(net)
        w0 = model(net, kind, 24); pz = pipe_zones(w0, cent); M0 = sim(w0, 'z0'); k = match(T24, M0)
        L = layouts(net, 8, 5, True, 500)
        for j in jobs:
            fp = f'{OUT}/zone_{net}_{kind}_{j}.json'
            if os.path.exists(fp): continue
            idx = L[j]; ob = obs_for(net, idx, 5000 + j)
            c0 = min(GRID, key=lambda c: np.sqrt(((B[c][:, k[idx]] - ob)**2).mean()))
            cache = {}
            def mis(Cz):
                key = tuple(Cz)
                if key not in cache:
                    h = win(sim(model(net, kind, 24, Cmap={p: Cz[z] for p, z in pz.items()}), f'z{net}{kind}{j}')['h'], 24).values
                    cache[key] = float(np.sqrt(((h[:, k[idx]] - ob)**2).mean()))
                return cache[key]
            Cz = [c0]*3; best = mis(Cz)
            for sweep in range(2):
                for z in range(3):
                    for v in ZGRID:
                        t = list(Cz); t[z] = v; e = mis(t)
                        if e < best - 1e-6: best, Cz = e, t
            r = metrics(T4, sim(model(net, kind, 4, Cmap={p: Cz[z] for p, z in pz.items()}), f'zp{net}{kind}{j}'), 4, k)
            rg = metrics(T4, sim(model(net, kind, 4, C=c0), f'zg{net}{kind}{j}'), 4, k)
            json.dump(dict(network=net, start=kind, layout=j, C_global=c0, C_zones=Cz, zone=r, global_same_layout=rg, sims=len(cache)), open(fp, 'w'), indent=1)
            print(net, kind, j, 'C0', c0, 'Cz', Cz, '| zone', {a: round(b, 1) for a, b in r.items() if a in ('head_rmse', 'p_rmse', 'short_found', 'false_alarms')},
                  '| global', {a: round(b, 1) for a, b in rg.items() if a in ('head_rmse', 'short_found', 'false_alarms')}, flush=True)
def bank4(net, kind):
    fp = f'{OUT}/_bank4_{net}_{kind}.pkl'; part = fp + '.part'
    if os.path.exists(fp): return pickle.load(open(fp, 'rb'))
    out = pickle.load(open(part, 'rb')) if os.path.exists(part) else {}
    for c in GRID:
        if c in out: continue
        out[c] = win(sim(model(net, kind, 4, C=c), f'b4{net}{kind}{c}')['h'], 4).values.astype(np.float32)
        pickle.dump(out, open(part, 'wb'))
    pickle.dump(out, open(fp, 'wb')); return out
if __name__ == '__main__' and sys.argv[1] == 'global4':
    net = sys.argv[2]; rows = []; T4 = truth(net, 4); pred = {}
    for kind in START[net]:
        B = bank4(net, kind); M0 = sim(model(net, kind, 4, C=GRID[0]), f'm4{net}'); k = match(T4, M0)
        for K in [1, 3, 5, 8]:
            for li, idx in enumerate(layouts(net, K, 20, False, 100 + K)):
                rng = np.random.default_rng(2000 + 37*K + li); h = win(T4['h'], 4).values[:, idx]; ob = h + rng.normal(0, NOISE, h.shape)
                c = min(GRID, key=lambda c: np.sqrt(((B[c][:, k[idx]] - ob)**2).mean()))
                if (kind, c) not in pred: pred[(kind, c)] = metrics(T4, sim(model(net, kind, 4, C=c), f'q{net}{kind}{c}'), 4, k)
                rows.append(dict(network=net, start=kind, method='global_supplyhours', K=K, layout=li, C=c, **pred[(kind, c)]))
        print(net, kind, 'global4 done', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/stageC_global4_{net}.csv', index=False)
