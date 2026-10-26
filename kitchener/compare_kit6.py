import wntr, numpy as np, pandas as pd, json
from scipy.spatial import cKDTree
def run(f):
    wn=wntr.network.WaterNetworkModel(f); r=wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=__import__('os').path.join(__import__('tempfile').gettempdir(), '') + f.split('/')[-1][:-4])
    J=wn.junction_name_list; xy=np.array([wn.get_node(j).coordinates for j in J])
    return wn,J,xy,r.node['pressure'][J],r.node['head'][J],r.link['velocity'].abs()
T=run('models/kit6_true.inp'); out={}; rows=[]
for v in ['class','uniform','design']:
    R=run(f'models/kit6_osm_{v}.inp'); tr=cKDTree(R[2]); d,k=tr.query(T[2])
    Rp=R[3].iloc[:,k].values; Tp=T[3].values; err=Rp-Tp
    pk=T[3].mean(axis=1).idxmin()   # peak (lowest-pressure) hour
    ipk=list(T[3].index).index(pk)
    e_pk=err[ipk]; e_all=err.ravel()
    lowT=set(np.where(T[3].min().values<45)[0]); lowR=set(np.where(R[3].min().values[k]<45)[0])
    res={'snap_dist_median_m':float(np.median(d)),'snap_dist_p90_m':float(np.percentile(d,90)),
      'RMSE_peak_m':float(np.sqrt((e_pk**2).mean())),'bias_peak_m':float(e_pk.mean()),'maxabs_peak_m':float(np.abs(e_pk).max()),
      'RMSE_24h_m':float(np.sqrt((e_all**2).mean())),
      'min_p_true':float(T[3].min().min()),'min_p_rebuild':float(R[3].min().min()),
      'vmax_true':float(T[5].max().max()),'vmax_rebuild':float(R[5].max().max()),
      'low45_true_n':len(lowT),'low45_rebuild_n':len(lowR),'low45_overlap':len(lowT&lowR),
      'peak_hour':int(pk/3600)}
    # elevation-mismatch part: elevation difference between snapped rebuild node and true node
    eT=np.array([T[0].get_node(j).elevation for j in T[1]]); eR=np.array([R[0].get_node(R[1][i]).elevation for i in k])
    res['RMSE_elev_snap_m']=float(np.sqrt(((eR-eT)**2).mean()))
    hd=R[4].iloc[:,k].values[ipk]-T[4].values[ipk]; res['RMSE_head_peak_m']=float(np.sqrt((hd**2).mean()))
    out[v]=res
    rows.append(pd.DataFrame({'node':T[1],'x':T[2][:,0],'y':T[2][:,1],'p_true':Tp[ipk],'p_rebuild':Rp[ipk],'err':e_pk,'snap_m':d,'variant':v}))
json.dump(out,open('kit6_compare.json','w'),indent=1)
pd.concat(rows).to_csv('kit6_compare_nodes.csv',index=False)
for v,r in out.items(): print(v,{k:round(x,2) for k,x in r.items()})
