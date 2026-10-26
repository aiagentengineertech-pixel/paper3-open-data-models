"""Open-data rebuild of a Lower Hutt DMA + comparison with the true model (same method as KIT 6).
Pipes follow OSM drivable streets inside the zone's convex hull +50 m; same feed location and head, DEM, buildings, pattern; C=130.
Variants: class (road-class sizes), design (class then upsize any pipe > 0.9 m/s at peak; standard sizes)."""
import sys, json, numpy as np, pandas as pd, wntr, rasterio, networkx as nx, collections
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, LineString, Polygon
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
ZONE = sys.argv[1]; CRS = 2193
g = json.load(open(f'{ZONE}_true_graph.json')); comp = g['comps'][0]
hull = MultiPoint(np.array([g['nodes'][str(i)] for i in comp])).convex_hull.buffer(50)
meta = json.load(open(f'models/{ZONE}_true_meta.json')); feedxy = [g['nodes'][str(f)] for f in meta['feeds']]
tb = Transformer.from_crs(4326, CRS, always_xy=True)
keep = {'primary':300,'secondary':300,'primary_link':300,'secondary_link':300,'tertiary':200,'tertiary_link':200,'residential':150,'unclassified':150,'living_street':150}
ways = [w for w in json.load(open('inputs/osm_streets.json'))['elements'] if w.get('tags',{}).get('highway') in keep]
cnt = collections.Counter(n for w in ways for n in w['nodes']); pos = {}; G = nx.Graph()
for w in ways:
    xs, ys = tb.transform([q['lon'] for q in w['geometry']], [q['lat'] for q in w['geometry']])
    for n, x, y in zip(w['nodes'], xs, ys): pos[n] = (x, y)
    D = keep[w['tags']['highway']]; seg = [w['nodes'][0]]
    for n in w['nodes'][1:]:
        seg.append(n)
        if cnt[n] > 1 or n == w['nodes'][-1]:
            L = LineString([pos[s] for s in seg])
            if L.length > 0 and hull.contains(L.interpolate(0.5, normalized=True)) and seg[0] != seg[-1]:
                a, b = seg[0], seg[-1]
                if not G.has_edge(a, b) or G[a][b]['L'] > L.length: G.add_edge(a, b, L=L.length, D=D, geom=[list(p) for p in L.coords])
            seg = [n]
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
ids = list(G.nodes); xy = np.array([pos[i] for i in ids]); tree = cKDTree(xy)
feeds = {ids[tree.query(f)[1]] for f in feedxy}
lo, la = Transformer.from_crs(CRS, 4326, always_xy=True).transform(xy[:,0], xy[:,1])
with rasterio.open('inputs/dem_glo30.tif') as r: elev = np.array([v[0] for v in r.sample(zip(lo, la))], float)
E = dict(zip(ids, elev)); dem = {i: 0.0 for i in ids}
for b in json.load(open('inputs/osm_buildings.json'))['elements']:
    geo = b.get('geometry') or []
    if len(geo) < 4: continue
    px, py = tb.transform([q['lon'] for q in geo], [q['lat'] for q in geo]); poly = Polygon(zip(px, py))
    if not poly.is_valid or poly.area < 25 or not hull.contains(poly.centroid): continue
    d, k = tree.query([poly.centroid.x, poly.centroid.y])
    if d <= 150: dem[ids[k]] += 0.65*min(20, max(1, poly.area/150))
def build(variant):
    wn = wntr.network.WaterNetworkModel(f'models/{ZONE}_true.inp')
    for l in list(wn.link_name_list): wn.remove_link(l)
    for n in list(wn.node_name_list): wn.remove_node(n)
    nm = lambda i: f'R{i}' if i in feeds else f'J{i}'
    for i in ids:
        if i in feeds: wn.add_reservoir(nm(i), base_head=meta['head'], coordinates=pos[i])
        else: wn.add_junction(nm(i), base_demand=dem[i]/86.4/1000, demand_pattern='diurnal', elevation=E[i], coordinates=pos[i])
    for k, (a, b, d) in enumerate(G.edges(data=True)):
        wn.add_pipe(f'P{k}', nm(a), nm(b), length=max(d['L'], 1), diameter=d['D']/1000, roughness=130)
    return wn
S = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.6, 0.75]
wn = build('class'); wntr.network.write_inpfile(wn, f'models/{ZONE}_osm_class.inp', units='LPS')
for it in range(30):
    v = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=__import__('os').path.join(__import__('tempfile').gettempdir(), 'lhsz')).link['velocity'].abs().max()
    bad = [l for l in wn.pipe_name_list if v[l] > 0.9]
    if not bad: break
    for l in bad:
        p = wn.get_link(l); i = min(range(len(S)), key=lambda j: abs(S[j]-p.diameter)); p.diameter = S[min(i+1, len(S)-1)]
wntr.network.write_inpfile(wn, f'models/{ZONE}_osm_design.inp', units='LPS')
json.dump({str(k): d['geom'] for k, (a, b, d) in enumerate(G.edges(data=True))}, open(f'models/{ZONE}_osm_geom.json', 'w'))
print(ZONE, 'rebuild nodes', len(ids), 'pipes', G.number_of_edges(), 'km %.1f' % (sum(d['L'] for *_, d in G.edges(data=True))/1000),
      'demand %.1f L/s' % (sum(dem.values())/86.4), 'design iters', it, sorted(collections.Counter(round(wn.get_link(l).diameter*1000) for l in wn.pipe_name_list).items()))
