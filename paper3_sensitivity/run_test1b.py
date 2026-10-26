"""Paper 3, Test 1b: (a) stress (demand x2) for friction inputs, (b) 15 seeds of elevation noise,
(c) spatially correlated + biased DEM-like error, (d) same error but control points surveyed (PRV outlets, tank, pump)."""
import wntr, numpy as np, pandas as pd, json, os, sys, time, pickle
from run_sensitivity import load, diam_default, rough_default, demand_even, BASE, THRESH
def scale_demand(wn, f):
    for j in wn.junction_name_list:
        for d in wn.get_node(j).demand_timeseries_list: d.base_value *= f
def elev_iid(wn, sd, seed):
    rng = np.random.default_rng(seed)
    for j in wn.junction_name_list: wn.get_node(j).elevation += rng.normal(0, sd)
def elev_corr(wn, sd, bias, L, seed, protect=()):
    js = wn.junction_name_list
    xy = np.array([wn.get_node(j).coordinates for j in js]); rng = np.random.default_rng(seed)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :])**2).sum(-1)); C = np.exp(-d / L) * sd**2
    e = np.linalg.cholesky(C + 1e-6 * np.eye(len(js))) @ rng.normal(size=len(js)) + bias
    for j, v in zip(js, e):
        if j not in protect: wn.get_node(j).elevation += v
def control_nodes(wn):
    s = set()
    for v in wn.valve_name_list: l = wn.get_link(v); s |= {l.start_node_name, l.end_node_name}
    for p in wn.pump_name_list: l = wn.get_link(p); s |= {l.start_node_name, l.end_node_name}
    for t in wn.tank_name_list:
        for l in wn.get_links_for_node(t): s |= {wn.get_link(l).start_node_name, wn.get_link(l).end_node_name}
    return s

def simulate(mods, tag):
    wn = load()
    for f in mods: f(wn)
    r = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(os.environ.get('TMP', '/tmp'), 'p3b_' + tag))
    return wn, r.node['pressure'][wn.junction_name_list], r.node['head']['T1']

def metrics(P, P0, T, T0, name):
    e = P - P0
    return dict(scenario=name, rmse_m=float(np.sqrt((e**2).values.mean())), bias_m=float(e.values.mean()),
        p95_abs_m=float(np.percentile(np.abs(e.values), 95)), max_abs_m=float(np.abs(e.values).max()),
        misclass_pct=float(((P0 < THRESH) != (P < THRESH)).values.mean() * 100),
        tank_rmse_m=float(np.sqrt(((T - T0)**2).mean())), min_p_true=float(P0.values.min()), min_p_model=float(P.values.min()))

if __name__ == '__main__':
    group = sys.argv[1]; out = []
    res = os.path.join(BASE, 'results', 'test1b'); os.makedirs(res, exist_ok=True)
    if group == 'stress':
        _, P0, T0 = simulate([lambda w: scale_demand(w, 2)], 'x2base')
        for name, m in {'X2_diam_trunk_only_100': [lambda w: diam_default(w, 200, 100)],
                        'X2_rough_all_C100': [lambda w: rough_default(w, 100)],
                        'X2_rough_all_C130': [lambda w: rough_default(w, 130)],
                        'X2_demand_even': [demand_even]}.items():
            _, P, T = simulate([lambda w: scale_demand(w, 2)] + m, name); out.append(metrics(P, P0, T, T0, name)); print(name, round(out[-1]['rmse_m'], 2), flush=True)
    else:
        b = pickle.load(open(os.path.join(BASE, 'results', '_baseline.pkl'), 'rb')); P0, T0 = b['p'], b['tank']
        if group.startswith('seeds'):
            a, z = map(int, group.split(':')[1].split('-'))
            for s in range(a, z):
                _, P, T = simulate([lambda w, s=s: elev_iid(w, 2.5, s)], f's{s}'); out.append(metrics(P, P0, T, T0, f'E2_seed{s}')); print(s, round(out[-1]['rmse_m'], 2), flush=True)
        elif group == 'corr':
            cn = control_nodes(load()); print('control nodes', sorted(cn))
            for name, m in {'EC1_corr_sd2.5_bias+1_L300': [lambda w: elev_corr(w, 2.5, 1.0, 300, 11)],
                            'EC2_same_controls_surveyed': [lambda w: elev_corr(w, 2.5, 1.0, 300, 11, protect=cn)],
                            'EI2_iid2.5_controls_surveyed': [lambda w: (lambda c: [w.get_node(j).__setattr__('elevation', w.get_node(j).elevation + (0 if j in c else np.random.default_rng(1000 + i).normal(0, 2.5))) for i, j in enumerate(w.junction_name_list)])(cn)],
                            'EI1_iid2.5_no_survey': [lambda w: [w.get_node(j).__setattr__('elevation', w.get_node(j).elevation + np.random.default_rng(1000 + i).normal(0, 2.5)) for i, j in enumerate(w.junction_name_list)]]}.items():
                _, P, T = simulate(m, name); out.append(metrics(P, P0, T, T0, name)); print(name, round(out[-1]['rmse_m'], 2), round(out[-1]['bias_m'], 2), flush=True)
    pd.DataFrame(out).to_csv(os.path.join(res, f"{group.replace(':', '_')}.csv"), index=False)
