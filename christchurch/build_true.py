"""Christchurch (CCC open data, CC BY 4.0): true EPANET model for one pressure zone.
Sourced: in-service Main/Submain/Trunk Main pipes whose midpoint lies in the zone polygon; nominal diameter (inside diameter where given);
material; reservoir base level and capacity. Assumed: reservoir water level = base + depth/2 (depth = capacity / footprint area);
reservoir joined to the nearest node on a main >= 190 mm by a main of the measured length at the zone's largest diameter;
H-W C by material; demands from OSM buildings (same rule as KIT 6 / Lower Hutt); same diurnal pattern; PE nominal = OD, ID = 0.82 x OD
where no inside diameter is given."""
import sys, os, json, numpy as np, networkx as nx, geopandas as gpd, wntr, rasterio
from shapely.geometry import Point, Polygon
from shapely.ops import substring
from scipy.spatial import cKDTree
from pyproj import Transformer
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ZONE, RES = sys.argv[1], sys.argv[2]; TOL = 3.0; tag = ZONE.replace(' ', '_')
z = gpd.read_file('pressure_zones.geojson').set_crs(2193, allow_override=True); poly = z[z.ZoneName == ZONE].geometry.union_all()
p = gpd.read_file(f'pipes_{tag}.geojson').set_crs(2193, allow_override=True)
p = p[p.ServiceStatus.eq('In Service') & p.Type.isin(['Main', 'Submain', 'Trunk Main', 'Crossover'])]
BUF = float(os.environ.get('ZBUF', '0')); p = p[p.geometry.centroid.within(poly.buffer(BUF))].explode(index_parts=False).reset_index(drop=True)
p['core'] = p.geometry.centroid.within(poly)
ends = np.array([g.coords[0][:2] for g in p.geometry] + [g.coords[-1][:2] for g in p.geometry]); et = cKDTree(ends); pieces = []
for i, r in p.iterrows():
    g = r.geometry; cut = []; minx, miny, maxx, maxy = g.bounds
    for k in et.query_ball_point([(minx+maxx)/2, (miny+maxy)/2], r=max(maxx-minx, maxy-miny)/2 + TOL + 1):
        pt = Point(ends[k])
        if g.distance(pt) < TOL:
            d = g.project(pt)
            if TOL < d < g.length - TOL: cut.append(d)
    pts = [0.0] + sorted(set(round(c, 2) for c in cut)) + [g.length]
    D = r.InsideDiameter if (r.InsideDiameter == r.InsideDiameter and r.InsideDiameter and r.InsideDiameter > 0) else None
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a >= 0.05: pieces.append((substring(g, a, b), r.NominalDiameter, D, r.Construction, r.Type, bool(r.core)))
vf = f'valves_{tag}.geojson'; nclosed = 0
if os.path.exists(vf):
    vv = gpd.read_file(vf).set_crs(2193, allow_override=True)
    cv = vv[(vv.NormalPosition == 'Closed') & (vv.ServiceStatus == 'In Service')]
    if len(cv):
        cvt = cKDTree(np.array([[g.x, g.y] for g in cv.geometry])); keep = []
        for q in pieces:
            mid = q[0].interpolate(0.5, normalized=True)
            near = cvt.query_ball_point([mid.x, mid.y], r=q[0].length/2 + 2)
            if any(q[0].distance(Point(cvt.data[j])) < 1.5 for j in near): nclosed += 1
            else: keep.append(q)
        pieces = keep
print('pieces cut at closed valves:', nclosed)
coords = np.array([q[0].coords[0][:2] for q in pieces] + [q[0].coords[-1][:2] for q in pieces])
U = nx.Graph(); U.add_nodes_from(range(len(coords))); U.add_edges_from(cKDTree(coords).query_pairs(TOL)); lab, cxy = {}, {}
for k, comp in enumerate(nx.connected_components(U)):
    for i in comp: lab[i] = k
    cxy[k] = coords[list(comp)].mean(axis=0)
n = len(pieces); G = nx.MultiGraph(); edges = []
for i, q in enumerate(pieces):
    a, b = lab[i], lab[i+n]
    if a != b: edges.append(dict(a=a, b=b, L=q[0].length, Dn=q[1], Di=q[2], mat=q[3], ptype=q[4], core=q[5], geom=[list(c[:2]) for c in q[0].coords]))
# bridging: keep core pipes; add shortest non-core paths (<=150 m) that join core components
C = nx.Graph(); [C.add_edge(e['a'], e['b']) for e in edges if e['core']]
F = nx.Graph()
for i, e in enumerate(edges):
    w = 0.0 if e['core'] else e['L']
    if not F.has_edge(e['a'], e['b']) or F[e['a']][e['b']]['w'] > w: F.add_edge(e['a'], e['b'], w=w, i=i)
keep = {i for i, e in enumerate(edges) if e['core']}; nbridge = 0
cc = sorted(nx.connected_components(C), key=len, reverse=True)
main = set(cc[0])
for other in cc[1:]:
    if len(other) < 3: continue
    try:
        L, path = nx.multi_source_dijkstra(F, list(other), weight='w')
        tgt = min((n for n in main if n in L), key=lambda n: L[n], default=None)
        if tgt is not None and L[tgt] <= 150:
            pth = path[tgt]
            for u, v in zip(pth[:-1], pth[1:]): keep.add(F[u][v]['i'])
            main |= set(other) | set(pth); nbridge += 1
    except nx.NetworkXNoPath: pass
edges = [e for i, e in enumerate(edges) if i in keep]
for e in edges: G.add_edge(e['a'], e['b'])
print('bridges added:', nbridge, ' non-core pipes kept:', sum(1 for e in edges if not e['core']))
comps = sorted(nx.connected_components(G), key=len, reverse=True); comp = comps[0]
print(ZONE, 'nodes', G.number_of_nodes(), 'edges', len(edges), 'components', len(comps), [len(c) for c in comps[:5]],
      'km largest %.1f of %.1f' % (sum(e['L'] for e in edges if e['a'] in comp)/1000, sum(e['L'] for e in edges)/1000))
json.dump(dict(nodes={int(k): list(map(float, v)) for k, v in cxy.items() if k in G}, edges=edges, comps=[sorted(map(int, c)) for c in comps]),
          open(f'{tag}_true_graph.json', 'w'), default=str)
# --- EPANET model ---
nodes = {k: cxy[k] for k in comp}; ids = sorted(nodes); xy = np.array([nodes[i] for i in ids]); tree = cKDTree(xy)
E_ = [e for e in edges if e['a'] in comp and e['b'] in comp]
rv = gpd.read_file('reservoirs.geojson').set_crs(2193, allow_override=True); rv = rv[rv.ReservoirName.fillna('').str.contains(RES) & rv.ServiceStatus.eq('In Service')]
depth = float((rv.ReservoirCapacity / rv.geometry.area).mean()); Hres = float(rv.BaseLevel.mean()) + depth/2
rc = rv.geometry.union_all().centroid
def _d(e):
    try: return float(e['Di'] or e['Dn'] or 0)
    except (TypeError, ValueError): return 0.0
bign = sorted({e['a'] for e in E_ if _d(e) >= 190} | {e['b'] for e in E_ if _d(e) >= 190})
bxy = np.array([nodes[i] for i in bign]); dfeed, kf = cKDTree(bxy).query([rc.x, rc.y]); feed = bign[kf]  # nearest node on a main >= 190 mm
lo, la = Transformer.from_crs(2193, 4326, always_xy=True).transform(xy[:, 0], xy[:, 1])
with rasterio.open('inputs/dem_glo30.tif') as r: elev = np.array([v[0] for v in r.sample(zip(lo, la))], float)
Ev = dict(zip(ids, elev)); tb = Transformer.from_crs(4326, 2193, always_xy=True); dem = {i: 0.0 for i in ids}; nb = 0
for b in json.load(open(f'inputs/osm_buildings_{tag}.json' if os.path.exists(f'inputs/osm_buildings_{tag}.json') else 'inputs/osm_buildings.json'))['elements']:
    geo = b.get('geometry') or []
    if len(geo) < 4: continue
    px, py = tb.transform([q['lon'] for q in geo], [q['lat'] for q in geo]); pg = Polygon(zip(px, py))
    if not pg.is_valid or pg.area < 25 or not poly.buffer(30).contains(pg.centroid): continue
    d, k = tree.query([pg.centroid.x, pg.centroid.y])
    if d <= 150: dem[ids[k]] += 0.65*min(20, max(1, pg.area/150)); nb += 1
wn = wntr.network.WaterNetworkModel(); wn.options.hydraulic.inpfile_units = 'LPS'; wn.options.hydraulic.headloss = 'H-W'
wn.options.time.duration = 24*3600; wn.options.time.hydraulic_timestep = wn.options.time.report_timestep = wn.options.time.pattern_timestep = 3600
wn.add_pattern('diurnal', [0.6,0.5,0.45,0.45,0.5,0.7,1.1,1.5,1.6,1.4,1.2,1.1,1.05,1.0,0.95,0.95,1.0,1.2,1.4,1.5,1.3,1.1,0.9,0.75])
for i in ids: wn.add_junction(f'J{i}', base_demand=dem[i]/86.4/1000, demand_pattern='diurnal', elevation=Ev[i], coordinates=tuple(nodes[i]))
wn.add_reservoir('RES', base_head=Hres, coordinates=(rc.x, rc.y))
def C(m):
    m = (m or '').upper()
    if 'PE' in m or 'PVC' in m: return 140
    if m in ('DI', 'CLDI', 'CLS'): return 130
    if m in ('AC', 'STEEL'): return 120
    if m in ('CI',): return 100
    if m == 'GALV': return 100
    return 130
nmiss = 0; Dmax = 0
for k, e in enumerate(E_):
    if e['Di']: D = float(e['Di'])
    else:
        try: D = float(e['Dn'])
        except (TypeError, ValueError): D = float('nan')
        if not np.isfinite(D) or D <= 0: D = 100.0; nmiss += 1
        elif 'PE' in str(e['mat']).upper(): D *= 0.82
    Dmax = max(Dmax, D)
    wn.add_pipe(f'P{k}', f'J{e["a"]}', f'J{e["b"]}', length=max(e['L'], 1.0), diameter=max(D, 10)/1000, roughness=C(e['mat']))
wn.add_pipe('FEED', 'RES', f'J{feed}', length=max(dfeed, 1.0), diameter=Dmax/1000, roughness=130)
wntr.network.write_inpfile(wn, f'models/{tag}_true.inp', units='LPS')
tot = sum(dem.values())/86.4
json.dump(dict(feeds=[feed], head=Hres, feed_len=float(dfeed), feed_D=Dmax, buildings=nb, avg_demand_Ls=tot, reservoir=RES), open(f'models/{tag}_true_meta.json', 'w'))
print(f'pipes {len(E_)} no-diam {nmiss}; buildings {nb}, demand {tot:.1f} L/s; elev {elev.min():.0f}-{elev.max():.0f} m; reservoir head {Hres:.1f} m (depth {depth:.1f}); feed main {dfeed:.0f} m, D {Dmax:.0f} mm')
