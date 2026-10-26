"""Paper 3, Test 1: which missing input matters most? (L-Town, 48 h, 5-min steps)
Each scenario damages ONE input the way a small utility would have to guess it,
then compares junction pressures, tank level and inflow with the true model.
"""
import wntr, numpy as np, pandas as pd, json, os, sys, time
BASE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(BASE, '..', 'epanet', 'Networks', 'L-TOWN.inp')
HOURS = 48; THRESH = 30.0; SEED = 7

def load():
    wn = wntr.network.WaterNetworkModel(INP)
    wn.options.time.duration = HOURS * 3600
    return wn

def diam_default(wn, keep_mm=200, default_mm=100):
    for p in wn.pipe_name_list:
        pp = wn.get_link(p)
        if pp.diameter * 1000 < keep_mm - 1: pp.diameter = default_mm / 1000
def rough_default(wn, c):
    for p in wn.pipe_name_list: wn.get_link(p).roughness = c
def demand_even(wn):
    js = wn.junction_name_list
    for k in range(3):
        tot = sum(wn.get_node(j).demand_timeseries_list[k].base_value for j in js)
        for j in js: wn.get_node(j).demand_timeseries_list[k].base_value = tot / len(js)
def demand_noise(wn, frac=0.5):
    rng = np.random.default_rng(SEED); js = wn.junction_name_list
    for k in range(3):
        b = np.array([wn.get_node(j).demand_timeseries_list[k].base_value for j in js])
        nb = b * rng.uniform(1 - frac, 1 + frac, len(b)); nb *= b.sum() / nb.sum()
        for j, v in zip(js, nb): wn.get_node(j).demand_timeseries_list[k].base_value = v
def elev_noise(wn, sd, bias=0.0):
    rng = np.random.default_rng(SEED)
    for j in wn.junction_name_list:
        wn.get_node(j).elevation += rng.normal(bias, sd)

SCEN = {
 'S0_baseline':            [],
 'D1_diam_trunk_only_100': [lambda w: diam_default(w, 200, 100)],
 'D2_diam_trunk_only_150': [lambda w: diam_default(w, 200, 150)],
 'R1_rough_all_C130':      [lambda w: rough_default(w, 130)],
 'R2_rough_all_C100':      [lambda w: rough_default(w, 100)],
 'Q1_demand_even':         [demand_even],
 'Q2_demand_noise_50pct':  [lambda w: demand_noise(w, 0.5)],
 'E1_elev_noise_1m':       [lambda w: elev_noise(w, 1.0)],
 'E2_elev_noise_2.5m':     [lambda w: elev_noise(w, 2.5)],
 'E3_elev_noise_5m':       [lambda w: elev_noise(w, 5.0)],
 'C1_combined_limited':    [lambda w: diam_default(w, 200, 100), lambda w: rough_default(w, 130), demand_even, lambda w: elev_noise(w, 2.5)],
}

def run(name, fns):
    wn = load()
    for f in fns: f(wn)
    wntr.network.write_inpfile(wn, os.path.join(BASE, 'models', name + '.inp'), units='CMH')
    r = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(os.environ.get('TMP', '/tmp'), 'p3_' + name))
    js = wn.junction_name_list
    return dict(p=r.node['pressure'][js], tank=r.node['head']['T1'] if 'T1' in r.node['head'] else None,
                inflow=r.link['flowrate'][[l for l in wn.link_name_list if l.upper().startswith('PUMP') or l.upper().startswith('PRV')]].abs().sum(axis=1) if True else None)

if __name__ == '__main__':
    import pickle
    want = sys.argv[1:] or [k for k in SCEN if k != 'S0_baseline']
    bp = os.path.join(BASE, 'results', '_baseline.pkl')
    if os.path.exists(bp): base = pickle.load(open(bp, 'rb'))
    else: base = run('S0_baseline', []); pickle.dump(base, open(bp, 'wb'))
    P0 = base['p']
    for name in want:
        t = time.time(); r = run(name, SCEN[name]); P = r['p']; Pb = P0
        err = (P - Pb); below0 = Pb < THRESH; below = P < THRESH
        row = dict(scenario=name,
            rmse_m=float(np.sqrt((err**2).values.mean())), bias_m=float(err.values.mean()),
            p95_abs_m=float(np.percentile(np.abs(err.values), 95)), max_abs_m=float(np.abs(err.values).max()),
            junction_hours_misclassified_pct=float((below0 != below).values.mean() * 100),
            junctions_low_true=int(below0.any().sum()), junctions_low_model=int(below.any().sum()),
            tank_rmse_m=float(np.sqrt(((r['tank'] - base['tank'])**2).mean())),
            min_p_true=float(Pb.values.min()), min_p_model=float(P.values.min()), run_s=round(time.time() - t, 1))
        json.dump(row, open(os.path.join(BASE, 'results', name + '.json'), 'w'))
        np.sqrt((err**2).mean()).rename(name).to_csv(os.path.join(BASE, 'results', name + '_pernode.csv'))
        print(name, round(row['rmse_m'], 2), flush=True)
    # aggregate whatever exists
    import glob
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(BASE, 'results', '*.json')))]
    pd.DataFrame(rows).to_csv(os.path.join(BASE, 'results', 'sensitivity_summary.csv'), index=False)
    cols = [pd.read_csv(f, index_col=0) for f in sorted(glob.glob(os.path.join(BASE, 'results', '*_pernode.csv')))]
    pd.concat(cols, axis=1).to_csv(os.path.join(BASE, 'results', 'sensitivity_pernode_rmse.csv'))
