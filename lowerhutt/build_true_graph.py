"""Lower Hutt (Hutt City Council / Wellington Water open data): build the true network graph for one DMA.
In-use mains, rider mains and trunk mains only (service/household connections, ducts excluded).
Lines split where another pipe's end touches them (<= 2 m), endpoints clustered within 2 m."""
import sys, geopandas as gpd, numpy as np, networkx as nx, json
from shapely.geometry import Point
from shapely.ops import substring
from scipy.spatial import cKDTree
ZONE = sys.argv[1]; TOL = 2.0
p = gpd.read_file('pipes_konini_delaney.geojson').set_crs(2193, allow_override=True)
z = p[(p.dma_zone == ZONE) & (p.operational_status == 'In Use') & p.pipe_type.isin(['Main', 'RIderMain', 'Trunk Main'])]
z = z.explode(index_parts=False).reset_index(drop=True)
ends = np.array([g.coords[0][:2] for g in z.geometry] + [g.coords[-1][:2] for g in z.geometry])
et = cKDTree(ends); pieces = []
for i, r in z.iterrows():
    g = r.geometry; cut = []
    minx, miny, maxx, maxy = g.bounds
    for k in et.query_ball_point([(minx+maxx)/2, (miny+maxy)/2], r=max(maxx-minx, maxy-miny)/2 + TOL + 1):
        pt = Point(ends[k])
        if g.distance(pt) < TOL:
            d = g.project(pt)
            if TOL < d < g.length - TOL: cut.append(d)
    pts = [0.0] + sorted(set(round(c, 2) for c in cut)) + [g.length]
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 0.05: continue
        pieces.append((substring(g, a, b), r.diameter_mm, r.material, r.asset_id, str(r.date_installed)[:4], r.pipe_type))
coords = np.array([q[0].coords[0][:2] for q in pieces] + [q[0].coords[-1][:2] for q in pieces])
U = nx.Graph(); U.add_nodes_from(range(len(coords))); U.add_edges_from(cKDTree(coords).query_pairs(TOL))
lab, cxy = {}, {}
for k, comp in enumerate(nx.connected_components(U)):
    for i in comp: lab[i] = k
    cxy[k] = coords[list(comp)].mean(axis=0)
n = len(pieces); G = nx.MultiGraph(); edges = []
for i, q in enumerate(pieces):
    a, b = lab[i], lab[i+n]
    if a == b: continue
    edges.append(dict(a=a, b=b, L=q[0].length, D=q[1], mat=q[2], wid=q[3], yr=q[4], ptype=q[5], geom=[list(c[:2]) for c in q[0].coords])); G.add_edge(a, b)
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print(ZONE, 'pieces', len(pieces), 'nodes', G.number_of_nodes(), 'edges', len(edges), 'components', len(comps), [len(c) for c in comps[:6]])
print('km in largest', round(sum(e['L'] for e in edges if e['a'] in comps[0])/1000, 1), 'of', round(sum(e['L'] for e in edges)/1000, 1))
json.dump(dict(nodes={int(k): list(map(float, v)) for k, v in cxy.items() if k in G}, edges=edges, comps=[sorted(map(int, c)) for c in comps]),
          open(f'{ZONE}_true_graph.json', 'w'), default=str)
