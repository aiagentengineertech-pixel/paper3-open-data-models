"""Open-data rebuild of KIT 6: pipes follow OSM drivable streets only.
Zone boundary = convex hull of KIT 6 mains (+50 m) (zone extents are known to a utility; layout is not used).
Feed locations, reservoir head, DEM, building demands and pattern identical to the true model.
Variants: 'class' = diameter by road class (secondary/primary 300, tertiary 200, other 150);
          'uniform' = all 150 mm.  C = 130 for all pipes.
"""
import json, sys, numpy as np, wntr, rasterio, networkx as nx
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, LineString, Point, Polygon
variant = sys.argv[1]; OUT = f'models/kit6_osm_{variant}.inp'
g = json.load(open('kit6_true_graph.json')); comp = g['comps'][0]
txy = np.array([g['nodes'][str(i)] for i in comp]); hull = MultiPoint(txy).convex_hull.buffer(50)
meta = json.load(open('models/kit6_true_meta.json'))
feedxy = [g['nodes'][str(f)] for f in meta['feeds']]
tb = Transformer.from_crs(4326, 26917, always_xy=True)
keep = {'primary':300,'secondary':300,'primary_link':300,'secondary_link':300,'tertiary':200,'tertiary_link':200,
        'residential':150,'unclassified':150,'living_street':150}
ways = [w for w in json.load(open('kit6_inputs/osm_streets.json'))['elements'] if w.get('tags',{}).get('highway') in keep]
cnt = {}
for w in ways:
    for n in w['nodes']: cnt[n] = cnt.get(n,0)+1
pos = {}; G = nx.Graph()
for w in ways:
    xs, ys = tb.transform([p['lon'] for p in w['geometry']], [p['lat'] for p in w['geometry']])
    for n,x,y in zip(w['nodes'],xs,ys): pos[n]=(x,y)
    D = keep[w['tags']['highway']]
    seg=[w['nodes'][0]]
    for n in w['nodes'][1:]:
        seg.append(n)
        if cnt[n]>1 or n==w['nodes'][-1]:
            L = LineString([pos[s] for s in seg])
            if L.length>0 and hull.contains(L.interpolate(0.5,normalized=True)):
                a,b=seg[0],seg[-1]
                if a!=b and (not G.has_edge(a,b) or G[a][b]['L']>L.length): G.add_edge(a,b,L=L.length,D=D)
            seg=[n]
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
ids = list(G.nodes); xy = np.array([pos[i] for i in ids]); tree = cKDTree(xy)
feeds = {ids[tree.query(f)[1]] for f in feedxy}
t = Transformer.from_crs(26917,4326,always_xy=True); lon,lat=t.transform(xy[:,0],xy[:,1])
with rasterio.open('kit6_inputs/dem_glo30.tif') as r: elev=np.array([v[0] for v in r.sample(zip(lon,lat))],float)
E=dict(zip(ids,elev)); dem={i:0.0 for i in ids}
for b in json.load(open('kit6_inputs/osm_buildings.json'))['elements']:
    geo=b.get('geometry') or []
    if len(geo)<4: continue
    px,py=tb.transform([p['lon'] for p in geo],[p['lat'] for p in geo]); poly=Polygon(zip(px,py))
    if not poly.is_valid or poly.area<25: continue
    if not hull.contains(poly.centroid): continue
    d,k=tree.query([poly.centroid.x,poly.centroid.y])
    if d>150: continue
    dem[ids[k]]+=0.65*min(20,max(1,poly.area/150))
wn=wntr.network.WaterNetworkModel('models/kit6_true.inp')  # copy options/pattern
for l in list(wn.link_name_list): wn.remove_link(l)
for n in list(wn.node_name_list): wn.remove_node(n)
nm=lambda i: f'R{i}' if i in feeds else f'J{i}'
for i in ids:
    if i in feeds: wn.add_reservoir(nm(i),base_head=meta['head'],coordinates=pos[i])
    else: wn.add_junction(nm(i),base_demand=dem[i]/86.4/1000,demand_pattern='diurnal',elevation=E[i],coordinates=pos[i])
for k,(a,b,d) in enumerate(G.edges(data=True)):
    wn.add_pipe(f'P{k}',nm(a),nm(b),length=max(d['L'],1),diameter=(d['D'] if variant=='class' else 150)/1000,roughness=130)
wntr.network.write_inpfile(wn,OUT,units='LPS')
print(variant,'nodes',len(ids),'pipes',G.number_of_edges(),'km %.1f'%(sum(d['L'] for *_,d in G.edges(data=True))/1000),'demand %.1f L/s'%(sum(dem.values())/86.4),'feeds',feeds)
