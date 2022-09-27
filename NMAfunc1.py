import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integrate
import math
import statistics as st
from sklearn.cross_decomposition import CCA
from sklearn.metrics import auc, r2_score, explained_variance_score, mean_squared_error

from LCC import *

t_ind_method=stats.ttest_ind
t_rel_method=stats.ttest_rel

brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],  # visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"],  # hippocampal
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP","TT"],  # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"],  # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"],  # basal ganglia
                ["BLA", "BMA", "EP", "EPd", "MEA"]  # cortical subplate
                ]

class researchParam:
  def __init__(self):
    self.param=dict()
    self.saved_trans=dict()

'''
  DATA EXTRACTORS
'''
def region_spks_extractor(sessions_data,area_index,a_p,pre_post_all,delay):
  '''
  Extracts data of neurons of region given by area_index across all sessions and returns 
  it as an array accompanied with a list of neurons present in these areas.
    sessions_data=> alldata
    area_index=> index no of desired brain_area in brain_groups list
    a_p=> 'a' for active trials
          'p' for passive trials
    pre_post_all=> -1 for pre-stimulus
                    0 for post-stimulus
                    1 for whole time bin
    delay=> no of time bin delay, 1 time bin corresponds to 10ms delay
    
  '''
  if a_p == 'a':
    spk='spks'
  else:
    spk='spks_passive'
  area_session_spks = list()
  session_areas = list()
  for s,session in enumerate(sessions_data):
    arr=int()
    areas_in_a_session = list()
    for area in range(session['brain_area'].shape[0]):
      if session['brain_area'][area] in brain_groups[area_index]:
        areas_in_a_session.append(session['brain_area'][area])
        t1 = [0+delay if pre_post_all == -1 or pre_post_all ==1 else 50+delay]
        t2 = [250 if pre_post_all == 0 or pre_post_all == 1 else 50]
        if isinstance(arr,int) == True:
          arr = session[spk][area,:,t1[0]:t2[0]].reshape(1,session[spk].shape[1],t2[0]-t1[0])
        else:
          arr = np.vstack((arr,session[spk][area,:,t1[0]:t2[0]].reshape(1,session[spk].shape[1],t2[0]-t1[0])))
    if isinstance(arr,int) == True:
      area_session_spks.append(np.zeros((0,0,0)))
    else:
      area_session_spks.append(arr)
    session_areas.append(areas_in_a_session)
  return area_session_spks, session_areas

def get_region_neurons_in_session_index(session_data,area_index):
  '''
  returns a list of all neurons present in region denoted by area_index in given session_data
  '''
  index_of_neurons_region=list()
  for area in range(session_data['brain_area'].shape[0]):
    if session_data['brain_area'][area] in brain_groups[area_index]:
      index_of_neurons_region.append(area)
  return index_of_neurons_region

def bi_region_spks_extractor(sessions_data,area1_index,area2_index,a_p,l=250):
  '''
  Returns two arrays consisting of neural data from regions denoted by area1_index, area2_index respectively 
  consisting of only instances of trials having valid recordings for both regions
    sessions_data=> alldata
    area1_index=> index no of 1st desired brain_area
    area2_index+> index no of 2nd desired brain area
    a_p=> 'a' for active trials
          'p' for passive trials
    l=> no of time bins to extract
    
  '''
  if a_p == 'a':
    spk='spks'
  else:
    spk='spks_passive'

  area1_session_spks =  list()
  area2_session_spks = list()
  
  for s,session in enumerate(sessions_data):
    area1_neurons_index = get_region_neurons_in_session_index(session,area1_index)
    area2_neurons_index = get_region_neurons_in_session_index(session,area2_index)

    if len(area1_neurons_index)==0 or len(area2_neurons_index)==0:
      arr1 = np.zeros((0,0,0))
      arr2 = np.zeros((0,0,0))
      pass
    else:
      tr=session[spk].shape[1]
      arr1=np.empty((0,tr,l))
      arr2=np.empty((0,tr,l))
      for i in area1_neurons_index:
        arr1 = np.vstack((arr1, session[spk][i,:,:].reshape(1,tr,l)))
      for i in area2_neurons_index:
        arr2 = np.vstack((arr2, session[spk][i,:,:].reshape(1,tr,l)))
    area1_session_spks.append(arr1)
    area2_session_spks.append(arr2)

  return area1_session_spks, area2_session_spks

'''
  STATISTICAL ANALYSIS, ANALYIS FUNCTIONS
'''
def success_integral_diff_time(dat):
  rows=13
  cols=3
  fig,ax=plt.subplots(nrows=rows, ncols=cols,figsize=(30,60))
  for r in range(0,rows):
    for c in range(0,cols):
      if (r*cols)+c>=39:
        break
      else:
        stimulus_class = np.zeros_like(dat[(r*cols)+c]['contrast_right'])
        stimulus_class[dat[(r*cols)+c]['contrast_right'] > dat[(r*cols)+c]['contrast_left']]= -1
        stimulus_class[dat[(r*cols)+c]['contrast_right'] < dat[(r*cols)+c]['contrast_left']]=1
        result = (stimulus_class==dat[(r*cols)+c]['response'])
        ax[r][c].plot(list(range(0,len(result))),np.cumsum(result))
        diff = np.diff(np.cumsum(result),1)*100
        ax[r][c].scatter(list(range(0,len(diff))),diff,marker='_')
        ax[r][c].plot(list(range(0,len(result))),list(range(0,len(result))),'--')
        ax[r][c].set_xlabel('trial')
        ax[r][c].set_ylabel('mean_correct_responses/correct_rate')
        ax[r][c].legend(['actual_responses_path','response_rate','correct_responses_path'])

def correlation_tests_results(func):
  def inner(**kwargs):
    assert len(kwargs['sessions_data1']) == len(kwargs['sessions_data2'])
    test_results_df=pd.DataFrame(columns=['values','score','session'])
    for s in range(len(kwargs['sessions_data1'])):
      kwargs['data1']=kwargs['sessions_data1'][s]
      kwargs['data2']=kwargs['sessions_data2'][s]
      result=func(**kwargs)
      test_results_df.loc[len(test_results_df)] = [result[0],'statistic','s'+str(s)]
      test_results_df.loc[len(test_results_df)] = [result[1],'pvalue','s'+str(s)]
    plt.figure(figsize=(30,10))
    sns.barplot(x='session', y='values', hue='score', data=test_results_df)
    return
  return inner

def class_correlator_plot(active_data,passive_data):
  c_values=pd.DataFrame(columns=['corr_coef','session'])
  rows=3
  cols=13
  for r in range(0,rows):
    for c in range(0,cols):
      if active_data[(r*cols)+c].shape != (0,):
        X = np.vstack((
            active_data[(r*cols)+c].T,
            passive_data[(r*cols)+c].T,
        ))
        c_values.loc[(r*cols)+c]=[np.corrcoef(X)[0,1], 's'+str((r*cols)+c)]
  plt.figure(figsize=(20,5))
  sns.barplot(x='session', y='corr_coef', data=c_values)

# ttest functions
def ttest(a,b,method):
  return method(a,b,nan_policy='omit')

@correlation_tests_results
def t_test(method,sessions_data1,sessions_data2,data1=0,data2=0,axis=0):
  assert data1.shape == data2.shape
  test = method(data1,data2)
  return [test[0],test[1]]

def regions_activity_ttest(regions_activity, method):
  region1 = merge_activity(regions_activity[0][0]) + merge_activity(regions_activity[0][1])
  region2 = merge_activity(regions_activity[1][0]) + merge_activity(regions_activity[1][1])
  return ttest(region1,region2,method)

def regions_activity_similarity_contest(region1_spks,region2_spks,delays,t0,t1,activity_method,ttest_method,t2=500,t3=2500):
  delay_list=[d for d in delays]
  region1_actv_pre_spks = partition_sessions_timebin(region1_spks[0],t0,t1)
  region1_psv_pre_spks = partition_sessions_timebin(region1_spks[1],t0,t1)

  region1_activity = activity_method(active_data=region1_actv_pre_spks, passive_data=region1_psv_pre_spks, plot=False)

  ttest_results=np.empty((0,2))

  for td in delay_list:
    actv_spks = partition_sessions_timebin(region2_spks[0],t0+td,t1+td)
    psv_spks = partition_sessions_timebin(region2_spks[1],t0+td,t1+td)

    activity = activity_method(active_data=actv_spks, passive_data=psv_spks, plot=False)

    ttest_results = np.vstack((
        ttest_results,
        regions_activity_ttest([region1_activity,activity], ttest_method)
        ))
  ttest_results=pd.DataFrame(ttest_results, columns=['statistics', 'p-value'])
  ttest_results['delays']=delay_list
  return ttest_results

def region_actv_psv_ttest(region_activity,method):
  actv=merge_activity(region_activity[0])
  psv=merge_activity(region_activity[1])
  return ttest(actv,psv,method)
  
def regions_activity_similarity_contest_pipeline(region1,region2,delays,t0,t1,ttest_method,t2,t3):
  research_param=researchParam()
  research_param.param['ei']=regions_activity_similarity_contest(
      region1,
      region2,
      delays,
      t0,t1,
      ei_analysis_pipeline,
      ttest_method,
      t2,t3
  )
  research_param.param['ei']['activity_measure']=['ei']*len(research_param.param['ei'])

  research_param.param['mean']=regions_activity_similarity_contest(
      region1,
      region2,
      delays,
      t0,t1,
      mean_features_pipeline,
      ttest_method,
      t2,t3
  )
  research_param.param['mean']['activity_measure']=['mean']*len(research_param.param['mean'])

  research_param.param['var']=regions_activity_similarity_contest(
      region1,
      region2,
      delays,
      t0,t1,
      var_features_pipeline,
      ttest_method,
      t2,t3
  )
  research_param.param['var']['activity_measure']=['var']*len(research_param.param['var'])

  research_param.param['peng']=regions_activity_similarity_contest(
      region1,
      region2,
      delays,
      t0,t1,
      peng_features_pipeline,
      ttest_method,
      t2,t3
  )
  research_param.param['peng']['activity_measure']=['peng']*len(research_param.param['peng'])

  return research_param

def regions_activity_similarity_contest_plot(research_param,y,plot_type):
  df = pd.concat(
      [research_param.param['ei'],research_param.param['mean']],
      ignore_index=True
  )
  df = pd.concat([df,research_param.param['var']], ignore_index=True)
  df = pd.concat([df,research_param.param['peng']], ignore_index=True)

  plt.figure(figsize=(25,8))
  plot_type(x='delays', y=y, hue='activity_measure', data=df)

#pearson correlaation test functions
def standardize(x):
  return (x-x.mean())/x.std()

def population_standardize(x):
  if x.ndim!=2:
    raise AssertionError('input data should be 2 dimensional')
  r=np.empty((0,x.shape[1]))
  for e in x:
    r=np.vstack((r,standardize(e)))
  return r

def combine_arr_in_list(l):
  l=sessions_averager(l,0)
  all_arr=0
  for e in l:
    if e.shape[0]==0:
      pass
    else:
      if isinstance(all_arr,int):
        all_arr=e
      else:
        all_arr=np.vstack((all_arr,e))
  return all_arr

def population_corr_coef(population):
  if len(population)!=2 or population[0].shape!=population[1].shape or population[0].ndim!=2 or population[1].ndim!=2:
    raise AssertionError('argument should be an iterable of two element 2D arrays of the same shape')
  ccoef=np.empty((0))
  for x,y in zip(population[0], population[1]):
    ccoef=np.hstack((ccoef,np.corrcoef(x,y)[0,1]))
  return ccoef

def bi_region_corr_coef(regions):
  region1 = combine_arr_in_list(regions[0])
  region2 = combine_arr_in_list(regions[1])

  return population_corr_coef([region1,region2])

'''
  SESSIONS PROCESSOR
'''
def sessions_averager(sessions_data,axis):
  sessions_average = list()
  for s,session in enumerate(sessions_data):
    if session.shape != (0,0,0):
      sessions_average.append(np.average(session,axis))
    else:
      sessions_average.append(np.zeros((0,0)))
  return sessions_average

#sessions features computer's
def sessions_feature_computer(func):
  def inner(**kwargs):
    sessions_feature = list()
    for s,session in enumerate(kwargs['sessions_data']):
      kwargs['data']=session
      sessions_feature.append(func(**kwargs))
    return sessions_feature
  return inner

@sessions_feature_computer
def mean_feature(sessions_data,axis=0,data=0):
  return np.mean(data,axis)

@sessions_feature_computer
def variance_feature(sessions_data,axis,data=0):
  return np.var(data,axis)

@sessions_feature_computer
def cumsum_mean_feature(sessions_data,axis,data=0):
  return np.mean(np.cumsum(data,axis),axis)

@sessions_feature_computer
def parserval_energy_feature(sessions_data,data=0,axis=0):
  return np.diagonal(data @ data.T)

@sessions_feature_computer
def entropy(sessions_data,data=0,base=2,axis=0):
  return stats.entropy(data)

# sessions engagement index functions
def ei_vector(active,passive):
  active = sessions_averager(active,1)
  passive = sessions_averager(passive,1)
  active = mean_feature(sessions_data=active,axis=0)
  passive = mean_feature(sessions_data=passive,axis=0)
  return [(a-p)/np.linalg.norm(a-p) for p,a in zip(passive,active)]

def sessions_ei_values(active_session, passive_session,vector_func):
  ei_vectors = vector_func(active_session, passive_session)
  active_ei_values=list()
  passive_ei_values=list()
  for s, ei_vector in enumerate(ei_vectors):
    active_ei_values.append([np.dot(trial,ei_vector) for trial in np.mean(active_session[s],0)])
    passive_ei_values.append([np.dot(trial,ei_vector) for trial in np.mean(passive_session[s],0)])
  return active_ei_values,passive_ei_values

def ei_plot(active_ei,passive_ei):
  rows=3
  cols=13

  fig,ax=plt.subplots(nrows=rows,ncols=cols,figsize=(30,10))

  for r in range(rows):
    for c in range(cols):
      actv=active_ei[(r*cols)+c]
      psv=passive_ei[(r*cols)+c]
      ax[r,c].scatter(list(range(len(actv))),actv,marker='.')
      ax[r,c].scatter(list(range(len(psv))),psv,marker='.')
      ax[r,c].legend(['active','passive'])

# sessions combinators
def combine_region_sessions_data(data,stein,label,l):
  X= np.empty((0,l))
  start=True
  for s,session in enumerate(data):
    if session.shape != (0,0):
      X = np.vstack((X,session))
      if start == True:
        if label == 1:
          y = stein[s]['feedback_type'].tolist()
        else:
          y = (np.ones((session.shape[0]))*label).tolist()
        start=False
      else:
        if label == 1:
          y = y + stein[s]['feedback_type'].tolist()
        else:
          y = y + (np.ones((session.shape[0]))*label).tolist()
  return np.array(X),y

def combine_region_sessions_feature(data,stein,label,l):
  X= np.empty((0))
  start=True
  for s,session in enumerate(data):
    if session.shape != (0,):
      X = np.hstack((X,session))
      if start == True:
        if label == 1:
          y = stein[s]['feedback_type'].tolist()
        else:
          y = (np.ones((session.shape[0]))*label).tolist()
        start=False
      else:
        if label == 1:
          y = y + stein[s]['feedback_type'].tolist()
        else:
          y = y + (np.ones((session.shape[0]))*label).tolist()
  return np.array(X),y

def merge_ei(active,passive):
  ei=list()
  for eis in active:
    ei = ei + eis
  for eis in passive:
    ei = ei + eis
  return ei

def merge_activity(sessions):
  '''
    return single list from merging of list of lists passed as sessions
  '''
  act=list()
  for s in sessions:
    act=act+s
  return act

#sessions timeframe extraction
def partition_session_timebin(session,t1,t2):
  t1 = int(t1 /10)
  t2 = int(t2/10)
  # print(t1,t2)
  return session[:,:,t1:t2]

def partition_sessions_timebin(stein,t1,t2):
  sessions = list()
  for session in stein:
    sessions.append(partition_session_timebin(session,t1,t2))
  return sessions

'''
  PIPELINES
'''
def features_pipeline(func):
  def inner(**kwargs):
    actv = sessions_averager(kwargs['active_data'],0)
    psv = sessions_averager(kwargs['passive_data'],0)

    kwargs['active_data'] = actv
    kwargs['passive_data'] = psv

    actv,psv = func(**kwargs)
    actv = [s.tolist() for s in actv]
    psv = [s.tolist() for s in psv]

    if kwargs['plot'] == True:
      ei_plot(actv, psv)
     
    return (actv,psv)
  return inner

@features_pipeline
def mean_features_pipeline(active_data, passive_data, plot=False):
  actv = mean_feature(sessions_data=active_data,axis=1)
  psv = mean_feature(sessions_data=passive_data,axis=1)
  return (actv,psv)

@features_pipeline
def var_features_pipeline(active_data, passive_data, plot=False):
  actv = variance_feature(sessions_data=active_data,axis=1)
  psv = variance_feature(sessions_data=passive_data,axis=1)
  return (actv,psv)

@features_pipeline
def peng_features_pipeline(active_data, passive_data, plot=False):
  actv = parserval_energy_feature(sessions_data=active_data,axis=1)
  psv = parserval_energy_feature(sessions_data=passive_data,axis=1)
  return (actv,psv)

def ei_analysis_pipeline(active_data,passive_data,plot=False):
  global ei_vector,ei_plot
  active_ei, passive_ei = sessions_ei_values(
    active_data,passive_data,ei_vector
  )
  if plot==True:
    ei_plot(active_ei, passive_ei)
  return (active_ei, passive_ei)

def get_sessions_feedback(active_data,passive_data,stein,length):
  global combine_region_sessions_data
  _, y1 = combine_region_sessions_data(active_data,stein,1,length)
  _, y2 = combine_region_sessions_data(passive_data,stein,0,length)
  return y1+y2

#CCA
def CCA_parameters_pipeline(active_data, passive_data, pre_length, post_active_data, post_passive_data, post_length, stein):
  global ei_vector
  #get sessions for pre-stimulus and post-stimulus data engagement_index
  active_ei, passive_ei = ei_analysis_pipeline(
    active_data,passive_data,plot=False
  )
  post_active_ei, post_passive_ei = ei_analysis_pipeline(
    post_active_data,post_passive_data,plot=False
  )

  #collapse sessions data across neurons
  active_TB, passive_TB = (
    sessions_averager(active_data,0),
    sessions_averager(passive_data,0)
  )

  ##features computation
  #mean feature
  active_TmeanB, passive_TmeanB = (
    mean_feature(
      sessions_data = active_TB,
      axis=1
    ),
    mean_feature(
      sessions_data = passive_TB,
      axis=1
    )
  )
  #variance feature
  active_TvarB, passive_TvarB = (
    variance_feature(
      sessions_data = active_TB,
      axis=1
    ),
    variance_feature(
      sessions_data = passive_TB,
      axis=1
    )
  )
  #parserval energy feature
  active_TenergyB, passive_TenergyB = (
    parserval_energy_feature(
      sessions_data = active_TB,
      axis=1
    ),
    parserval_energy_feature(
      sessions_data = passive_TB,
      axis=1
    )
  )

  #X variant construction
  M = np.hstack((
    combine_region_sessions_feature(active_TmeanB,stein,1,1)[0],
    combine_region_sessions_feature(passive_TmeanB,stein,0,1)[0]
  ))
  V = np.hstack((
    combine_region_sessions_feature(active_TvarB,stein,1,1)[0],
    combine_region_sessions_feature(passive_TvarB,stein,0,1)[0]
  ))
  E = np.hstack((
    combine_region_sessions_feature(active_TenergyB,stein,1,1)[0],
    combine_region_sessions_feature(passive_TenergyB,stein,0,1)[0]
  ))
  MVE = np.array([M,V,E]).T
  # MVE=0

  #Y variant construction
  y = get_sessions_feedback(active_TB,passive_TB,stein,pre_length)
  pre_ei = merge_ei(active_ei, passive_ei)
  post_ei = merge_ei(post_active_ei, post_passive_ei)
  # Y = np.array([y,pre_ei,post_ei]).T
  Y = np.array([pre_ei]).T
  # Y=0

  return MVE,Y

def bi_region_CCA_analysis(stein,actv,psv,t1,t2):

  '''
    actv[0]=bg_actv
    actv[1]=or_actv
    psv[0]=bg_psv
    psv[1]=or_psv
    t1=delay start
    t2=delay end
  '''

  global partition_sessions_timebin,partition_session_timebin
  #bg spks extraction
  bg_actv_pre_spks = partition_sessions_timebin(actv[0],50,250)
  bg_psv_pre_spks = partition_sessions_timebin(psv[0],50,250)
  bg_actv_post_spks = partition_sessions_timebin(actv[0],500,2500)
  bg_psv_post_spks = partition_sessions_timebin(psv[0],500,2500)

  #or spks extraction
  or_actv_pre_spks_d = partition_sessions_timebin(actv[1],t1,t2)
  or_psv_pre_spks_d = partition_sessions_timebin(psv[1],t1,t2)
  or_actv_post_spks = partition_sessions_timebin(actv[1],500,2500)
  or_psv_post_spks = partition_sessions_timebin(psv[1],500,2500)

  ##region variant construction
  #bg region
  bg_MVE, bg_Y = CCA_parameters_pipeline(
  bg_actv_pre_spks,
  bg_psv_pre_spks,
  20,
  bg_actv_post_spks,
  bg_psv_post_spks,
  200,
  stein
  )
  bg = np.hstack((bg_MVE,bg_Y))
  #or region
  or_d_MVE, or_d_Y = CCA_parameters_pipeline(
      or_actv_pre_spks_d,
      or_psv_pre_spks_d,
      20,
      or_actv_post_spks,
      or_psv_post_spks,
      200,
      stein
  )
  or_ = np.hstack((or_d_MVE, or_d_Y))
  
  ##CCA implementation
  bg_or_cca = CCA(6,scale=False)
  bg_or_cca.fit(bg, or_)
  Y_pred = bg_or_cca.predict(bg)
  XY_transfd = bg_or_cca.transform(bg, or_)
  R_squared = bg_or_cca.score(bg, or_)
  mean_R_squared = r2_score(or_[:,0], Y_pred[:,0])
  var_R_squared = r2_score(or_[:,1], Y_pred[:,1])
  energy_R_squared = r2_score(or_[:,2], Y_pred[:,2])
  pre_ei_R_squared = r2_score(or_[:,3], Y_pred[:,3])

  return bg,or_,Y_pred,R_squared,mean_R_squared,var_R_squared,energy_R_squared,pre_ei_R_squared

# pipeline result analysis
# def cca_cor_plot(bg_or_pred,bg,or_,Y_pred=None):
#   if bg_or_pred==1:
#     cca_df = pd.DataFrame({
#       'or_mean_c': Y_pred[:,0],
#       'or_var_c': Y_pred[:,1],
#       'or_energy_c': Y_pred[:,2],
#       'or_pre_ei_c': Y_pred[:,4],
#       'or_post_ei_c': Y_pred[:,5],
#       'bg_mean': bg[:,0],
#       'bg_var': bg[:,1],
#       'bg_energy': bg[:,2],
#       'bg_pre_ei': bg[:,4],
#       'bg_post_ei': bg[:,5]
#   })
#   else:
#     cca_df = pd.DataFrame({
#         'or_mean': or_[:,0],
#         'or_var': or_[:,1],
#         'or_energy': or_[:,2],
#         'or_pre_ei': or_[:,4],
#         'or_post_ei': or_[:,5],
#         'bg_mean': bg[:,0],
#         'bg_var': bg[:,1],
#         'bg_energy': bg[:,2],
#         'bg_pre_ei': bg[:,4],
#         'bg_post_ei': bg[:,5]
#     })
  
  # corr_cca_df=cca_df.corr(method='pearson')

  # plt.figure(figsize=(15,12))
  # cca_df_lt = corr_cca_df.where(np.tril(np.ones(corr_cca_df.shape)).astype(np.bool))
  # sns.heatmap(cca_df_lt,cmap="coolwarm",annot=True,fmt='.1g')
  # plt.tight_layout()
  # plt.savefig("Heatmap_Canonical_Correlates_from_X_and_data.jpg",
  #                     format='jpeg',
  #                     dpi=100)

#pearson
def bi_region_time_pearson_test_pipeline(region1,region2,delays,std,plot,t0=50,t1=250,t2=500,t3=2500):
  research_param=researchParam()

  delay_list=[d for d in delays]
  pearson=pd.DataFrame(columns=['delay','pearson','a_p'])
  pearson_stats=pd.DataFrame(columns=['delay','var','mean','h_quantile','99pile','30pile','20pile','10pile','a_p'])
  for td in delay_list:
    rgn1_actv=partition_sessions_timebin(region1[0],t0+td,t1+td)
    rgn2_actv=partition_sessions_timebin(region2[0],t0+td,t1+td)
    rgn1_psv=partition_sessions_timebin(region1[1],t0+td,t1+td)
    rgn2_psv=partition_sessions_timebin(region2[1],t0+td,t1+td)

    actv_r=bi_region_corr_coef([rgn1_actv,rgn2_actv]).tolist()
    a=['actv']*len(actv_r)
    d=[td]*len(actv_r)
    psv_r=bi_region_corr_coef([rgn1_psv,rgn2_psv]).tolist()
    p=['psv']*len(psv_r)
    d = d + [td]*len(psv_r)
    df=pd.DataFrame(columns=['delay','area','a_p'])
    df['delay']=d
    df['pearson']=actv_r+psv_r
    df['a_p']=a+p
    cdf=pd.DataFrame(columns=['delay','pearson','a_p'])
    pearson = pd.concat([pearson,df])

    avar=np.nanvar(actv_r)
    pvar=np.nanvar(psv_r)
    amean=np.nanmean(actv_r)
    pmean=np.nanmean(psv_r)
    a_qtile,a_99pile,a_30pile, a_20pile, a_10pile = (np.nanquantile(actv_r,0.5),np.nanpercentile(actv_r,99),np.nanpercentile(actv_r,30), np.nanpercentile(actv_r,20), np.nanpercentile(actv_r,10))
    p_qtile,p_99pile,p_30pile, p_20pile, p_10pile = (np.nanquantile(psv_r,0.5),np.nanpercentile(psv_r,99),np.nanpercentile(psv_r,30), np.nanpercentile(psv_r,20), np.nanpercentile(psv_r,10)) 
    pearson_stats.loc[len(pearson_stats)]=[td, avar,amean,a_qtile,a_99pile,a_30pile, a_20pile, a_10pile,'actv']
    pearson_stats.loc[len(pearson_stats)]=[td, pvar,pmean,p_qtile,p_99pile,p_30pile, p_20pile, p_10pile,'psv']


  # rgn1_actv=partition_sessions_timebin(region1[0],t2,t3)
  # rgn2_actv=partition_sessions_timebin(region2[0],t2,t3)
  # rgn1_psv=partition_sessions_timebin(region1[1],t2,t3)
  # rgn2_psv=partition_sessions_timebin(region2[1],t2,t3)

  # actv_r=bi_region_corr_coef([rgn1_actv,rgn2_actv]).tolist()
  # a=['actv']*len(actv_r)
  # d=[td+10]*len(actv_r)
  # psv_r=bi_region_corr_coef([rgn1_psv,rgn2_psv]).tolist()
  # p=['psv']*len(psv_r)
  # d = d + [td+10]*len(psv_r)
  # df=pd.DataFrame(columns=['delay','area','a_p'])
  # df['delay']=d
  # df['pearson']=actv_r+psv_r
  # df['a_p']=a+p
  # cdf=pd.DataFrame(columns=['delay','pearson','a_p'])
  # pearson = pd.concat([pearson,df])

  # avar=np.nanvar(actv_r)
  # pvar=np.nanvar(psv_r)
  # amean=np.nanmean(actv_r)
  # pmean=np.nanmean(psv_r)
  # a_qtile,a_99pile,a_30pile, a_20pile, a_10pile = (np.nanquantile(actv_r,0.5),np.nanpercentile(actv_r,99),np.nanpercentile(actv_r,30), np.nanpercentile(actv_r,20), np.nanpercentile(actv_r,10))
  # p_qtile,p_99pile,p_30pile, p_20pile, p_10pile = (np.nanquantile(psv_r,0.5),np.nanpercentile(psv_r,99),np.nanpercentile(psv_r,30), np.nanpercentile(psv_r,20), np.nanpercentile(psv_r,10)) 
  # pearson_stats.loc[len(pearson_stats)]=[td+10, avar,amean,a_qtile,a_99pile,a_30pile, a_20pile, a_10pile,'actv']
  # pearson_stats.loc[len(pearson_stats)]=[td+10, pvar,pmean,p_qtile,p_99pile,p_30pile, p_20pile, p_10pile,'psv']


  research_param.param['pearson']=pearson
  research_param.param['pearson_stats']=pearson_stats

  research_param.vplot=sns.violinplot
  research_param.bplot=sns.barplot
  research_param.stpplot=sns.stripplot
  research_param.sctplot=sns.scatterplot
  research_param.ptplot=sns.pointplot
  research_param.saved_trans['pearson_distb']={'x':'delay','y':'pearson','hue':'a_p','data':research_param.param['pearson']}
  research_param.saved_trans['pearson_distb_stats']={'x':'delay','y':'var','hue':'a_p','data':research_param.param['pearson_stats']}

  return research_param

'''
  TO BE ARRANGED LATER
'''
def bi_region_lcc_pipeline(r1, r2, delays,t0=50,t1=250,t2=500,t3=2500):

  research_param=researchParam()

  delay_list=[d for d in delays]

  linearity_df=pd.DataFrame(columns=['delay','linearity'])

  for td in delay_list:
    re1=np.vstack((
      combine_arr_in_list(partition_sessions_timebin(r1[0],t0+td,t1+td)),
      combine_arr_in_list(partition_sessions_timebin(r1[1],t0+td,t1+td))
    ))
    re2=np.vstack((
      combine_arr_in_list(partition_sessions_timebin(r2[0],t0+td,t1+td)),
      combine_arr_in_list(partition_sessions_timebin(r2[1],t0+td,t1+td))
    ))
    lcc=LCC_param([re1,re2])
    linearity_df.loc[len(linearity_df)]=[td,lcc._linearity]

  # re1=np.vstack((
  #   combine_arr_in_list(partition_sessions_timebin(r1[0],t2,t3)),
  #   combine_arr_in_list(partition_sessions_timebin(r1[1],t2,t3))
  # ))
  # re2=np.vstack((
  #   combine_arr_in_list(partition_sessions_timebin(r2[0],t2,t3)),
  #   combine_arr_in_list(partition_sessions_timebin(r2[1],t2,t3))
  # ))
  # lcc=LCC_param([re1,re2])
  # linearity_df.loc[len(linearity_df)]=[td+10,lcc._linearity]

  research_param.param['linearity']=linearity_df

  return  research_param
  
def bi_region_CCA_pipeline(stein,r1, r2, delays,t0=50,t1=250,t2=500,t3=2500):
  research_param=researchParam()
  research_param.param['delays']=[d for d in delays]

  delay_list=[d for d in delays]

  research_param.param['window_shift_CCA']=np.empty((4,4,len(delay_list)))
  research_param.param['score']=np.empty((5,len(delay_list)))

  for ti,td in enumerate(delay_list):
    re1,re2,pred,R_squared,mean,var,energy,pre_ei = bi_region_CCA_analysis(stein, (r1[0],r2[0]), (r1[1],r2[1]),t0+td,t1+td)

    for i in range(0,4):
      for j in range(0,4):
        research_param.param['window_shift_CCA'][i,j,ti] = np.corrcoef(re1[:,i],pred[:,j])[0,1]

    research_param.param['score'][0,ti]=R_squared
    research_param.param['score'][1,ti]=mean
    research_param.param['score'][2,ti]=var
    research_param.param['score'][3,ti]=energy
    research_param.param['score'][4,ti]=pre_ei

  research_param.param['score']= pd.DataFrame(research_param.param['score'].T,columns=['R_squared','mean','var','energy','pre_ei'])
  research_param.param['score']['delay']= [d for d in delay_list]
  
  return research_param

def CCA_cluster_plot(research,delays):
  for i in range(research.shape[0]):
    sns.clustermap(pd.DataFrame(research[i,:,:], index=['mean','var','energy','pre_ei'], columns=delays),annot=True, figsize=(20,6))

def CCA_metric_plot(df,R=False):
  plt.figure(figsize=(15,5))
  if R==True:
    sns.lineplot(x='delay',y='R_squared',data=df,color='hotpink')
  sns.lineplot(x='delay',y='mean',data=df,color='indigo')
  sns.lineplot(x='delay',y='var',data=df,color='steelblue')
  sns.lineplot(x='delay',y='energy',data=df,color='mediumseagreen')
  sns.lineplot(x='delay',y='pre_ei',data=df,color='dimgrey')
  if R==True:
    plt.legend(['R_squared','mean','var','energy','pre_ei'])
  else:
    plt.legend(['mean','var','energy','pre_ei'])