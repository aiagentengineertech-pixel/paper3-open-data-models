"""'design' variant: start from road-class diameters, then iteratively upsize any pipe whose peak-hour
velocity exceeds 0.9 m/s to the next standard size (150,200,250,300,400,450,600 mm). Uses only public-data
model + a generic design criterion (no utility pipe records)."""
import wntr, numpy as np
S=[0.15,0.2,0.25,0.3,0.4,0.45,0.6,0.75]
wn=wntr.network.WaterNetworkModel('models/kit6_osm_class.inp')
for it in range(30):
    r=wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=__import__('os').path.join(__import__('tempfile').gettempdir(), 'kd')); v=r.link['velocity'].abs().max()
    bad=[l for l in wn.pipe_name_list if v[l]>0.9]
    if not bad: break
    for l in bad:
        p=wn.get_link(l); i=min(range(len(S)),key=lambda j:abs(S[j]-p.diameter)); p.diameter=S[min(i+1,len(S)-1)]
print('iterations',it,'remaining >0.9',len(bad))
import collections; print(sorted(collections.Counter(round(wn.get_link(l).diameter*1000) for l in wn.pipe_name_list).items()))
wntr.network.write_inpfile(wn,'models/kit6_osm_design.inp',units='LPS')
