from NMAfunc1 import *
no_of_sessions=39
brain_areas_label = ['v', 't', 'h', 'n', 'mb', 'g', 'c']
t_method=0

def region_activity_bulldozer(active_data, passive_data, axis):
  active = [sessions_averager(active_data[area],axis) for area,_ in enumerate(brain_groups)]
  passive = [sessions_averager(passive_data[area],axis) for area,_ in enumerate(brain_groups)]
  return active, passive

def region_mean_feature(active_data,passive_data,axis):
  active = [mean_feature(sessions_data=active_data[area],axis=axis) for area,_ in enumerate(brain_groups)]
  passive = [mean_feature(sessions_data=passive_data[area],axis=axis) for area,_ in enumerate(brain_groups)]
  return active,passive

def regions_correlate_estimate_plotter(func):
  def inner(**kwargs):
    rows = 3
    cols = 13
    fig,ax=plt.subplots(nrows=rows,ncols=cols,figsize=(70,20))
    for r in range(rows):
      for c in range(cols):
        kwargs['session'] = (r*cols)+c
        kwargs['plotter_func'](
              func(**kwargs),
              ax0=ax[r,c],
              xticks=brain_areas_label[kwargs['areas'][0]:kwargs['areas'][1]],
              yticks=brain_areas_label[kwargs['areas'][0]:kwargs['areas'][1]],
          )
    fig.tight_layout()
    plt.show
    return
  return inner

@regions_correlate_estimate_plotter
def regions_corrcoef(data=0,session=0,l=0,areas=(0,-1),plotter_func=0):
  assert session < no_of_sessions
  for area,_ in enumerate(brain_groups):
    if data[area][session].shape == (0,):
      data[area][session] = np.zeros((l))
  data = [data[area][session] for area,_ in enumerate(brain_groups)]
  corr = np.corrcoef(np.array(data))
  return corr[areas[0]:areas[-1], :]

@regions_correlate_estimate_plotter
def regions_ttest(method=0,data=0,session=0,l=0,areas=(0,-1),plotter_func=0,s_p=0):
  assert session < no_of_sessions
  for area,_ in enumerate(brain_groups):
    if data[area][session].shape == (0,):
      data[area][session] = np.zeros((l))
  data = [data[area][session] for area,_ in enumerate(brain_groups)]
  statistic=list()
  pvalue=list()
  for area in data:
    area_stat=list()
    area_pval=list()
    for area2 in data:
      result=method(area,area2,0)
      area_stat.append(result[0])
      area_pval.append(result[1])
    statistic.append(area_stat)
    pvalue.append(area_pval)
  ttest = [statistic, pvalue]
  return ttest[s_p]

def heatmap(corr=0,ax0=0,xticks=0,yticks=0):
  sns.heatmap(corr,xticklabels=xticks,yticklabels=yticks,ax=ax0)
  return


def neuron_sorter(data,neuron_labels,wanted_neuron,l):
  neuron_data=np.empty((0,l))
  for s,session in enumerate(data):
    for n,neuron in enumerate(session):
      if neuron_labels[s][n] == wanted_neuron:
        neuron_data = np.vstack((neuron_data,data[s][n]))
  return neuron_data

def region_neuron_sorter(spks,neuron_id,a,l):
  region_data=list()
  region_neuron_id=list()
  for nn in brain_groups[a]:
    data=neuron_sorter(spks,neuron_id,nn,l)
    if data.shape[0]!=0:
      region_data.append(data)
      region_neuron_id.append(nn)
  return region_data, region_neuron_id

def region_neuron_average_plot(actv,psv,neurons):
  rows = 2
  cols = 5
  fig, ax= plt.subplots(nrows=rows,ncols=cols,figsize=(30,10))
  for r in range(rows):
    for c in range(cols):
      n=(r*cols)+c
      actv_average = np.mean(actv[n],0)
      psv_average = np.mean(psv[n],0)
      ax[r,c].scatter(list(range(actv_average.shape[0])),actv_average,marker='|')
      ax[r,c].scatter(list(range(psv_average.shape[0])),psv_average,marker='|')
      ax[r,c].set_title(neurons[n])
      ax[r,c].legend(['active','passive'])