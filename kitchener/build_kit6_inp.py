"""Build the 'true' KIT 6 EPANET model from Kitchener open-data mains.
Assumptions (documented for Paper 3):
- Largest connected component of KIT 6 mains (kit6_true_graph.json comps[0]).
- Feeds: fixed-head reservoirs at the two trunk ends on the south boundary (600 mm node 816, 450 mm node 809).
  Head = highest node elevation + 35 m (zone HGL not published).
- Elevations: Copernicus GLO-30 DEM sampled at node locations.
- Demands: OSM building footprints, centroid -> nearest node (<=150 m);
  0.65 m3/d per building scaled by footprint/150 m2 (capped 1..20). 24 h diurnal pattern.
- Roughness (H-W C): PVC/PVCO/HDPE 140, DI 130, CPP 120. Minor losses ignored.
"""
import json, numpy as np, wntr, rasterio, geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, LineString
import sys
OUT = sys.argv[1] if len(sys.argv)>1 else 'models/kit6_true.inp'
g = json.load(open('kit6_true_graph.json'))
comp = set(g['comps'][0])
nodes = {int(k): v for k, v in g['nodes'].items() if int(k) in comp}
edges = [e for e in g['edges'] if e['a'] in comp and e['b'] in comp and e['a'] != e['b']]
ids = sorted(nodes); xy = np.array([nodes[i] for i in ids])
# elevations
t = Transformer.from_crs(26917, 4326, always_xy=True)
lon, lat = t.transform(xy[:,0], xy[:,1])
with rasterio.open('kit6_inputs/dem_glo30.tif') as r:
    elev = np.array([v[0] for v in r.sample(zip(lon, lat))], dtype=float)
E = dict(zip(ids, elev))
# demands from buildings
tb = Transformer.from_crs(4326, 26917, always_xy=True)
bl = json.load(open('kit6_inputs/osm_buildings.json'))['elements']
tree = cKDTree(xy); dem = {i: 0.0 for i in ids}; nb = 0
for b in bl:
    geo = b.get('geometry') or []
    if len(geo) < 4: continue
    px, py = tb.transform([p['lon'] for p in geo], [p['lat'] for p in geo])
    poly = Polygon(zip(px, py))
    if not poly.is_valid or poly.area < 25: continue   # skip sheds
    d, k = tree.query([poly.centroid.x, poly.centroid.y])
    if d > 150: continue
    dem[ids[k]] += 0.65 * min(20, max(1, poly.area/150)); nb += 1
# m3/d -> L/s
wn = wntr.network.WaterNetworkModel()
wn.options.hydraulic.inpfile_units = 'LPS'; wn.options.hydraulic.headloss = 'H-W'
wn.options.time.duration = 24*3600; wn.options.time.hydraulic_timestep = 3600
wn.options.time.report_timestep = 3600; wn.options.time.pattern_timestep = 3600
pat = [0.6,0.5,0.45,0.45,0.5,0.7,1.1,1.5,1.6,1.4,1.2,1.1,1.05,1.0,0.95,0.95,1.0,1.2,1.4,1.5,1.3,1.1,0.9,0.75]
wn.add_pattern('diurnal', pat)
feeds = [816, 809]; H = float(max(elev)) + 35
for i in ids:
    if i in feeds:
        wn.add_reservoir(f'R{i}', base_head=H, coordinates=tuple(nodes[i]))
    else:
        wn.add_junction(f'J{i}', base_demand=dem[i]/86.4/1000, demand_pattern='diurnal',
                        elevation=E[i], coordinates=tuple(nodes[i]))
C = {'PVC':140,'PVCO':140,'HDPE':140,'DI':130,'CPP':120}
nm = lambda i: f'R{i}' if i in feeds else f'J{i}'
for k, e in enumerate(edges):
    wn.add_pipe(f'P{k}_{e["wid"]}', nm(e['a']), nm(e['b']), length=max(e['L'],1.0),
                diameter=e['D']/1000, roughness=C.get(e['mat'],130))
wntr.network.write_inpfile(wn, OUT, units='LPS')
tot = sum(dem.values())/86.4
print(f'nodes {len(ids)} pipes {len(edges)} buildings {nb} demand avg {tot:.1f} L/s peak {tot*max(pat):.1f} L/s')
print(f'elev {elev.min():.1f}-{elev.max():.1f} m, reservoir head {H:.1f} m')
json.dump({'feeds':feeds,'head':H,'buildings':nb,'avg_demand_Ls':tot}, open(OUT.replace('.inp','_meta.json'),'w'))
