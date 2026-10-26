"""True EPANET model for a Lower Hutt DMA from Wellington Water / Hutt City open data.
Sourced: pipe routes, diameters, materials (HCC 3 Waters asset data); reservoir levels (Konini Street TWL 181.01 m, BWL 173.71 m).
Assumed: fixed-head reservoir at mid level (TWL+BWL)/2; reservoir connects to the nearest network node;
PE internal diameter = 0.82 x nominal (SDR11); H-W C by material (PE/PVC 140, CU/DI 130, AC/STCL 120, RCON 110, GI 100);
demands from OSM buildings exactly as KIT 6 (0.65 m3/d per building x footprint/150 m2, capped 1-20); same diurnal pattern."""
import sys, json, numpy as np, wntr, rasterio, geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
ZONE, TANK = sys.argv[1], sys.argv[2]
g = json.load(open(f'{ZONE}_true_graph.json')); comp = set(g['comps'][0])
nodes = {int(k): v for k, v in g['nodes'].items() if int(k) in comp}
edges = [e for e in g['edges'] if e['a'] in comp and e['b'] in comp and e['a'] != e['b']]
ids = sorted(nodes); xy = np.array([nodes[i] for i in ids]); tree = cKDTree(xy)
t = gpd.read_file('tanks.geojson').set_crs(2193, allow_override=True); tk = t[t.alias == TANK].iloc[0]
H = (float(tk.twl_m) + float(tk.bwl_m)) / 2; feed = ids[tree.query([tk.geometry.x, tk.geometry.y])[1]]
dfeed = tree.query([tk.geometry.x, tk.geometry.y])[0]
tl = Transformer.from_crs(2193, 4326, always_xy=True); lon, lat = tl.transform(xy[:,0], xy[:,1])
with rasterio.open('inputs/dem_glo30.tif') as r: elev = np.array([v[0] for v in r.sample(zip(lon, lat))], float)
E = dict(zip(ids, elev))
tb = Transformer.from_crs(4326, 2193, always_xy=True); dem = {i: 0.0 for i in ids}; nb = 0
for b in json.load(open('inputs/osm_buildings.json'))['elements']:
    geo = b.get('geometry') or []
    if len(geo) < 4: continue
    px, py = tb.transform([q['lon'] for q in geo], [q['lat'] for q in geo]); poly = Polygon(zip(px, py))
    if not poly.is_valid or poly.area < 25: continue
    d, k = tree.query([poly.centroid.x, poly.centroid.y])
    if d > 150: continue
    dem[ids[k]] += 0.65 * min(20, max(1, poly.area/150)); nb += 1
wn = wntr.network.WaterNetworkModel()
wn.options.hydraulic.inpfile_units = 'LPS'; wn.options.hydraulic.headloss = 'H-W'
for a in ['duration']: wn.options.time.duration = 24*3600
wn.options.time.hydraulic_timestep = wn.options.time.report_timestep = wn.options.time.pattern_timestep = 3600
pat = [0.6,0.5,0.45,0.45,0.5,0.7,1.1,1.5,1.6,1.4,1.2,1.1,1.05,1.0,0.95,0.95,1.0,1.2,1.4,1.5,1.3,1.1,0.9,0.75]
wn.add_pattern('diurnal', pat)
for i in ids:
    if i == feed: wn.add_reservoir(f'R{i}', base_head=H, coordinates=tuple(nodes[i]))
    else: wn.add_junction(f'J{i}', base_demand=dem[i]/86.4/1000, demand_pattern='diurnal', elevation=E[i], coordinates=tuple(nodes[i]))
def C(m):
    m = (m or '').upper()
    if m.startswith('PE') or 'PVC' in m: return 140
    if m in ('CU', 'DI', 'DICL'): return 130
    if m in ('AC', 'STCL', 'ST'): return 120
    if m == 'RCON': return 110
    if m == 'GI': return 100
    return 130
nm = lambda i: f'R{i}' if i == feed else f'J{i}'
nmiss = [0]
for k, e in enumerate(edges):
    try: D = float(e['D'])
    except (TypeError, ValueError): D = float('nan')
    if not np.isfinite(D) or D <= 0: D = 100.0; nmiss[0] += 1
    D = D*0.82 if str(e['mat']).upper().startswith('PE') else D
    wn.add_pipe(f'P{k}', nm(e['a']), nm(e['b']), length=max(e['L'], 1.0), diameter=max(D, 15)/1000, roughness=C(e['mat']))
out = f'models/{ZONE}_true.inp'; wntr.network.write_inpfile(wn, out, units='LPS')
tot = sum(dem.values())/86.4
print('pipes with no diameter (set 100 mm):', nmiss[0]); print(f'nodes {len(ids)} pipes {len(edges)} buildings {nb} demand avg {tot:.1f} L/s; elev {elev.min():.0f}-{elev.max():.0f} m; head {H:.1f} m; feed node {feed} ({dfeed:.0f} m from tank)')
json.dump({'feeds': [feed], 'head': H, 'buildings': nb, 'avg_demand_Ls': tot, 'tank': TANK}, open(out.replace('.inp', '_meta.json'), 'w'))
