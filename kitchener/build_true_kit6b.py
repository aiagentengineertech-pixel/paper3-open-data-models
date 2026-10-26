import geopandas as gpd, numpy as np, networkx as nx, json
from shapely.ops import split, snap, unary_union, linemerge
from shapely.geometry import Point, MultiPoint, LineString
m = gpd.read_file('kitchener/kitchener_water.gpkg', layer='water_mains')
z = m[m.PRESSURE_ZONE == 'KIT 6'].explode(index_parts=False).reset_index(drop=True)
TOL = 2.0
ends = [Point(g.coords[0]) for g in z.geometry] + [Point(g.coords[-1]) for g in z.geometry]
sidx = z.sindex
pieces = []
for i, r in z.iterrows():
    g = r.geometry
    cut = []
    for p in ends:
        if g.distance(p) < TOL:
            d = g.project(p)
            if TOL < d < g.length - TOL: cut.append(d)
    ds = sorted(set(round(c, 2) for c in cut))
    pts = [0.0] + ds + [g.length]
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 0.05: continue
        from shapely.ops import substring
        pieces.append((substring(g, a, b), r.PIPE_SIZE, r.MATERIAL, int(r.WATMAINID), r.INSTALLATION_DATE))
def key(c): return (round(c[0]/TOL)*TOL, round(c[1]/TOL)*TOL)
# cluster endpoints within TOL
coords = np.array([p[0].coords[0] for p in pieces] + [p[0].coords[-1] for p in pieces])
from scipy.spatial import cKDTree
t = cKDTree(coords); pairs = t.query_pairs(TOL)
U = nx.Graph(); U.add_nodes_from(range(len(coords))); U.add_edges_from(pairs)
lab = {}; cxy = {}
for k, comp in enumerate(nx.connected_components(U)):
    c = coords[list(comp)].mean(axis=0)
    for i in comp: lab[i] = k
    cxy[k] = c
n = len(pieces)
G = nx.MultiGraph()
edges = []
for i, p in enumerate(pieces):
    a, b = lab[i], lab[i + n]
    if a == b: continue
    edges.append(dict(a=a, b=b, L=p[0].length, D=p[1], mat=p[2], wid=p[3], yr=str(p[4])[:4], geom=[list(c) for c in p[0].coords]))
    G.add_edge(a, b)
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print('pieces', len(pieces), 'nodes', G.number_of_nodes(), 'edges', len(edges), 'components', len(comps), [len(c) for c in comps[:6]])
print('km in largest', round(sum(e['L'] for e in edges if e['a'] in comps[0]) / 1000, 1))
json.dump(dict(nodes={int(k): list(map(float, v)) for k, v in cxy.items() if k in G}, edges=edges, comps=[sorted(map(int, c)) for c in comps]),
          open('kitchener/kit6_true_graph.json', 'w'), default=str)
