"""Paper 3 v1.2 figures (10 Oct 2026). Run from core_protocol/: python3 figs_v12/make_figs_v12.py <outdir> [maps]"""
import os, sys, json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.abspath('.')); from core import *
O = sys.argv[1]; os.makedirs(O, exist_ok=True); R = 'results'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
  'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
  'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'savefig.dpi': 300})
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; GREY = '#b5b4ad'; INK = '#0b0b0b'; INK2 = '#52514e'
NETS = [('LTOWN', 'L-Town'), ('KIT6', 'Kitchener KIT 6'), ('CHCH', 'Christchurch Mt Pleasant 3'), ('LHUTT', 'Lower Hutt Konini Parkway')]
A = pd.read_csv(f'{R}/stageA_all.csv')
INP = {'LTOWN': None, 'KIT6': ('../kitchener/models/kit6_true.inp', '../kitchener/models/kit6_osm_class.inp'),
       'CHCH': ('../christchurch/models/Mt_Pleasant_3_true.inp', '../christchurch/models/Mt_Pleasant_3_osm_class.inp'),
       'LHUTT': ('../lowerhutt/models/Konini_Parkway_true.inp', '../lowerhutt/models/Konini_Parkway_osm_class.inp')}
def segs(wn):
    return [(wn.get_link(p).start_node.coordinates, wn.get_link(p).end_node.coordinates) for p in wn.link_name_list]
def draw(ax, wn, c, lw, z):
    for a, b in segs(wn): ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, zorder=z, solid_capstyle='round')
def scalebar(ax, km):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); ax.set_ylim(y0 - 0.12*(y1-y0), y1); y0, y1 = ax.get_ylim()
    L = km*1000; x = x0 + 0.02*(x1-x0); y = y0 + 0.03*(y1-y0)
    ax.plot([x, x+L], [y, y], color=INK, lw=2); ax.text(x+L/2, y+0.02*(y1-y0), f'{km:g} km', ha='center', va='bottom', fontsize=8, color=INK)
# ---------- Fig 1 networks
def fig1():
    f, axs = plt.subplots(2, 2, figsize=(7.2, 7.2))
    for ax, (k, name) in zip(axs.flat, NETS):
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        if k == 'LTOWN':
            wn = NET['LTOWN']['true'](); draw(ax, wn, S[0], 0.6, 2)
            ax.set_title(f'(a) {name}\n(schematic coordinates, no scale)', fontsize=9, loc='left')
        else:
            t = wntr.network.WaterNetworkModel(INP[k][0]); r = wntr.network.WaterNetworkModel(INP[k][1])
            draw(ax, t, S[0], 0.6, 2); draw(ax, r, S[1], 1.1, 1)
            for n in t.reservoir_name_list: x, y = t.get_node(n).coordinates; ax.plot(x, y, 's', ms=7, mfc=INK, mec='white', zorder=5)
            lab = {'KIT6': '(b)', 'CHCH': '(c)', 'LHUTT': '(d)'}[k]; ax.set_title(f'{lab} {name}', fontsize=9, loc='left')
            ax.set_aspect('equal'); scalebar(ax, 1 if k != 'KIT6' else 2)
        if k == 'LTOWN': ax.set_aspect('equal')
    h = [plt.Line2D([], [], color=S[0], lw=1.5), plt.Line2D([], [], color=S[1], lw=1.5), plt.Line2D([], [], marker='s', ls='', mfc=INK, mec='white', ms=7)]
    f.legend(h, ['Utility network (ground truth)', 'Rebuild from OpenStreetMap streets', 'Supply point'], loc='lower center', ncol=3, fontsize=8.5)
    f.tight_layout(rect=(0, 0.04, 1, 1)); f.savefig(f'{O}/Fig1_networks.png'); plt.close(f)
# ---------- Fig 2 workflow
def fig2():
    f, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    def box(x, y, w, h, t, fc='#f3f2ee', ec=INK2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=0.9))
        ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=7.2, color=INK, fontweight='bold' if bold else 'normal', wrap=True)
    def arr(x0, y0, x1, y1): ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9))
    box(0.1, 3.5, 2.4, 1.2, 'True network\n(benchmark, or published\nutility pipe records)')
    box(0.1, 1.6, 2.4, 1.2, 'Public data only\nOpenStreetMap streets\nand buildings, 30 m\nelevation model')
    box(3.0, 3.5, 2.6, 1.2, 'Test A\nguess one input at a time:\nroughness, pipe sizes,\nground level, demand spread')
    box(3.0, 1.6, 2.6, 1.2, 'Test B\nrebuild along streets;\nroad-class or design-rule\npipe sizes')
    box(6.3, 2.5, 2.3, 1.3, 'Test C\n1-8 simulated pressure\nloggers; fit roughness\n(network or 3 districts)')
    box(1.2, 0.1, 7.6, 1.0, 'Every model is run under continuous supply and under four hours of supply per day, and compared\nwith the truth on head and pressure error, under-supplied junctions found, and false alarms', fc='#ffffff')
    arr(2.5, 4.1, 3.0, 4.1); arr(2.5, 2.2, 3.0, 2.2); arr(5.6, 4.0, 6.3, 3.4); arr(5.6, 2.2, 6.3, 2.9); arr(7.45, 2.5, 7.45, 1.1)
    ax.text(9.4, 3.15, 'loggers read on\na normal day or\nduring supply\nhours', ha='center', va='center', fontsize=7.5, color=INK2)
    f.tight_layout(); f.savefig(f'{O}/Fig2_workflow.png'); plt.close(f)
# ---------- Fig 3 Test A pressure error
INPUTS = [('elev_noise_2.5m', 'Ground\nlevel'), ('rough_C100', 'Roughness'), ('diam_lt200_to_100', 'Pipe\nsizes'), ('demand_even', 'Demand\nspread'), ('all_guessed', 'All four')]
def fig3():
    f, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True); w = 0.19; x = np.arange(len(INPUTS))
    for ax, N, ttl in zip(axs, [24, 4], ['(a) Continuous supply', '(b) Four hours of supply per day']):
        for i, (k, name) in enumerate(NETS):
            v = [A[(A.network == k) & (A.model == m) & (A.supply_h == N)].p_rmse.iloc[0] for m, _ in INPUTS]
            ax.bar(x + (i-1.5)*w, v, w*0.9, color=S[i], label=name)
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in INPUTS]); ax.set_title(ttl, loc='left', fontsize=9)
    axs[0].set_ylabel('Pressure error, root-mean-square (m)'); axs[0].legend(fontsize=7.5, loc='upper left')
    f.tight_layout(); f.savefig(f'{O}/Fig3_testA_error.png'); plt.close(f)
# ---------- Fig 4 Test A false alarms
def fig4():
    f, ax = plt.subplots(figsize=(7.2, 3.0)); nets = [n for n in NETS if n[0] != 'KIT6']; w = 0.26; x = np.arange(len(INPUTS))
    cols = {'LTOWN': S[0], 'CHCH': S[2], 'LHUTT': S[3]}
    for i, (k, name) in enumerate(nets):
        d = A[(A.network == k) & (A.supply_h == 4)]; tr = int(d.short_true.iloc[0])
        v = [d[d.model == m].false_alarms.iloc[0] for m, _ in INPUTS]
        b = ax.bar(x + (i-1)*w, v, w*0.9, color=cols[k], label=f'{name} ({tr} truly under-supplied)')
        for xx, vv in zip(x + (i-1)*w, v): ax.text(xx, vv + 5, f'{vv:.0f}', ha='center', va='bottom', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([n.replace('\n', ' ') for _, n in INPUTS]); ax.set_ylabel('False alarms (junctions)')
    ax.legend(fontsize=7.5, loc='upper left'); f.tight_layout(); f.savefig(f'{O}/Fig4_testA_false_alarms.png'); plt.close(f)
# ---------- Fig 5 maps of under-supply
def cls(T, M, k):
    qT, rT = win(T['q'], 4).sum().values, win(T['req'], 4).sum().values; qM, rM = win(M['q'], 4).sum().values, win(M['req'], 4).sum().values
    with np.errstate(invalid='ignore', divide='ignore'): dT = qT/rT; dM = (qM/rM)[k]
    ok = np.isfinite(dT) & np.isfinite(dM); lT, lM = (dT < 0.9) & ok, (dM < 0.9) & ok; return lT & lM, lT & ~lM, ~lT & lM
def fig5():
    f, axs = plt.subplots(2, 2, figsize=(7.2, 7.4)); out = {}
    for row, (net, C) in enumerate([('CHCH', 170), ('LHUTT', 180)]):
        T = truth(net, 4); wnT = NET[net]['true']()
        for col, (lab, kw) in enumerate([('no loggers', {}), ('5 loggers, supply hours', {'C': C})]):
            M = sim(model(net, 'all_guessed', 4, **kw), f'f5{net}{col}'); hit, miss, fa = cls(T, M, np.arange(len(T['names'])))
            ax = axs[row, col]; ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
            draw(ax, wnT, '#d9d8d2', 0.6, 1); xy = T['xy']
            ax.scatter(xy[hit, 0], xy[hit, 1], s=9, c=S[0], marker='o', lw=0, zorder=3)
            ax.scatter(xy[miss, 0], xy[miss, 1], s=14, c=S[2], marker='s', lw=0, zorder=4)
            ax.scatter(xy[fa, 0], xy[fa, 1], s=16, c=S[1], marker='^', lw=0, zorder=5)
            nm = 'Christchurch' if net == 'CHCH' else 'Lower Hutt'
            ax.set_title(f'({"abcd"[row*2+col]}) {nm}, {lab}\nfound {hit.sum()} of {hit.sum()+miss.sum()}, false alarms {fa.sum()}', fontsize=8, loc='left')
            ax.set_aspect('equal'); scalebar(ax, 0.5 if net == 'CHCH' else 1); out[f'{net}_{col}'] = [int(hit.sum()), int(miss.sum()), int(fa.sum())]
    h = [plt.Line2D([], [], marker='o', ls='', mfc=S[0], mec=S[0], ms=5), plt.Line2D([], [], marker='s', ls='', mfc=S[2], mec=S[2], ms=5), plt.Line2D([], [], marker='^', ls='', mfc=S[1], mec=S[1], ms=6)]
    f.legend(h, ['Under-supplied, found', 'Under-supplied, missed', 'False alarm'], loc='lower center', ncol=3, fontsize=8.5)
    f.tight_layout(rect=(0, 0.04, 1, 1)); f.savefig(f'{O}/Fig5_undersupply_maps.png'); plt.close(f); json.dump(out, open(f'{O}/fig5_counts.json', 'w'))
# ---------- Fig 6 tuning
SUM = pd.read_csv(f'{R}/core_summary_stageC.csv')
ROWS = [('LTOWN', 'all_guessed', 'L-Town\nall guessed'), ('CHCH', 'all_guessed', 'Christchurch\nall guessed'), ('CHCH', 'osm_class', 'Christchurch\nrebuild'),
        ('LHUTT', 'all_guessed', 'Lower Hutt\nall guessed'), ('LHUTT', 'osm_class', 'Lower Hutt\nrebuild,\nroad class'), ('LHUTT', 'osm_design', 'Lower Hutt\nrebuild,\ndesign rule')]
def fig6():
    f, ax = plt.subplots(figsize=(7.2, 3.4)); w = 0.2; x = np.arange(len(ROWS))
    M = [('none', 'No loggers'), ('day', 'Five loggers, normal day'), ('supply', 'Five loggers, supply hours'), ('zone', 'Eight loggers, three districts')]
    for i, (m, lab) in enumerate(M):
        fa = []; fo = []
        for net, st, _ in ROWS:
            r = SUM[(SUM.network == net) & (SUM.start == st)].iloc[0]; fa.append(r[f'{m}_fa']); fo.append(r[f'{m}_found'])
        ax.bar(x + (i-1.5)*w, fa, w*0.9, color=S[i], label=lab)
        for xx, a, b in zip(x + (i-1.5)*w, fa, fo): ax.text(xx, a + 8, f'{b:.0f}', ha='center', va='bottom', fontsize=6.5, color=INK2, rotation=90)
    tr = [SUM[(SUM.network == n) & (SUM.start == s)].short_true.iloc[0] for n, s, _ in ROWS]
    ax.set_xticks(x); ax.set_xticklabels([f'{l}\n({t} true)' for (_, _, l), t in zip(ROWS, tr)], fontsize=7.5)
    ax.set_ylabel('False alarms (junctions)'); ax.set_ylim(0, 450); ax.legend(fontsize=7.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.13))
    f.tight_layout(); f.savefig(f'{O}/Fig6_tuning.png'); plt.close(f)
# ---------- Fig 7 fitted roughness
def fig7():
    f, ax = plt.subplots(figsize=(7.2, 3.3)); starts = [('LTOWN', 'all_guessed', 'L-Town\nall guessed'), ('KIT6', 'all_guessed', 'Kitchener\nall guessed'), ('KIT6', 'osm_design', 'Kitchener\nrebuild,\ndesign rule'),
        ('KIT6', 'osm_class', 'Kitchener\nrebuild,\nroad class'), ('CHCH', 'all_guessed', 'Christchurch\nall guessed'), ('CHCH', 'osm_class', 'Christchurch\nrebuild'),
        ('LHUTT', 'all_guessed', 'Lower Hutt\nall guessed'), ('LHUTT', 'osm_class', 'Lower Hutt\nrebuild,\nroad class'), ('LHUTT', 'osm_design', 'Lower Hutt\nrebuild,\ndesign rule')]
    ax.axhspan(100, 140, color='#eceae4', zorder=0); ax.axhline(60, color=INK2, ls='--', lw=0.8); ax.axhline(230, color=INK2, ls='--', lw=0.8)
    ax.text(len(starts)-0.45, 120, 'material values\nused in the true\nmodels (100-140)', fontsize=7, color=INK2, va='center', ha='right')
    ax.text(-0.45, 233, 'upper search limit', fontsize=7, color=INK2, va='bottom'); ax.text(-0.45, 63, 'lower search limit', fontsize=7, color=INK2, va='bottom')
    rng = np.random.default_rng(1)
    for i, (net, st, _) in enumerate(starts):
        for j, (fn, col, mk, dx) in enumerate([(f'{R}/stageC_global_{net}.csv', S[0], 'o', -0.15), (f'{R}/stageC_global4_{net}.csv', S[1], 'D', 0.15)]):
            d = pd.read_csv(fn); d = d[(d.start == st) & (d.K == 5)].C.values
            ax.scatter(i + dx + rng.uniform(-0.06, 0.06, len(d)), d, s=12, c=col, marker=mk, lw=0, alpha=0.8, zorder=3)
            ax.plot([i+dx-0.1, i+dx+0.1], [np.median(d)]*2, color=INK, lw=1.4, zorder=4)
    ax.set_xticks(range(len(starts))); ax.set_xticklabels([s[2] for s in starts], fontsize=7); ax.set_ylim(40, 250)
    ax.set_ylabel('Fitted Hazen-Williams coefficient'); ax.set_xlim(-0.5, len(starts)-0.5)
    h = [plt.Line2D([], [], marker='o', ls='', mfc=S[0], mec=S[0], ms=5), plt.Line2D([], [], marker='D', ls='', mfc=S[1], mec=S[1], ms=5), plt.Line2D([], [], color=INK, lw=1.4)]
    ax.legend(h, ['Loggers read on a normal day', 'Loggers read during supply hours', 'Median of 20 layouts'], fontsize=7.5, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.12))
    f.tight_layout(); f.savefig(f'{O}/Fig7_fitted_roughness.png'); plt.close(f)
for fn in sys.argv[2:]: globals()[fn](); print(fn, 'ok', flush=True)
