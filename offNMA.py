import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import LCC

class research:
  def __init__(self):
    self.parameters=dict()
    self.transformers=dict()
    self.methods=dict()

sessions_params_file=open('data_results/sessions_params.sav','rb')
sessions_params=pickle.load(sessions_params_file)
sessions_params_file.close()


'''
  BASIC OPERATIONS
'''
def array2D_standardize(data,axis=0):
  print(data.shape)
  return (data-np.repeat(np.mean(data,axis).reshape(data.shape[0],1),data.shape[1],1))/np.repeat(np.std(data,axis).reshape(data.shape[0],1),data.shape[1],1)

'''
    NEURAL DATA OPERATIONS
'''
def check_valid_sessions(sessions):
  count=0
  indexes=list()
  if isinstance(sessions[0],list):
    for s,session in enumerate(sessions):
      if session!=[]:
        indexes.append(s)

  return len(indexes),indexes

def stack_arrays_in_list(data_list=None):
    arr=np.array([])
    for data in data_list:
        if data.size!=0:
            if arr.size==0:
                arr=data
            else:
                arr=np.vstack((arr,data))
    return arr


'''
  ANALYSIS
'''

def success_integral_diff_time(dat):
  rows=13
  cols=3
  fig,ax=plt.subplots(nrows=rows, ncols=cols,figsize=(30,60),sharey='row')
  for r in range(0,rows):
    for c in range(0,cols):
      if (r*cols)+c>=39:
        break
      else:
        stimulus_class = np.zeros_like(dat[(r*cols)+c]['contrast_right'])
        stimulus_class[dat[(r*cols)+c]['contrast_right'] > dat[(r*cols)+c]['contrast_left']]= -1
        stimulus_class[dat[(r*cols)+c]['contrast_right'] < dat[(r*cols)+c]['contrast_left']]=1
        stimulus_class[dat[(r*cols)+c]['contrast_right'] == dat[(r*cols)+c]['contrast_left']]=0
        result = (stimulus_class==dat[(r*cols)+c]['response'])
        ax[r][c].plot(list(range(0,len(result))),np.cumsum(result))
        diff = np.diff(np.cumsum(result),1)*100
        ax[r][c].scatter(list(range(0,len(diff))),diff,marker='|')
        ax[r][c].plot(list(range(0,len(result))),list(range(0,len(result))),'--')
        # ax[r][c].set_xlabel('trial')
        # ax[r][c].set_ylabel('mean_correct_responses/correct_rate')
        ax[r][c].legend(['actual_responses_path','100=correct_response','correct_responses_path'])
        ax[r][c].set_title(dat[(r*cols)+c]['mouse_name']+' '+dat[(r*cols)+c]['date_exp'])
    ax[0][0].set_xlabel('trial')
    ax[0][0].set_ylabel('mean_correct_responses/correct_rate')
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.2, wspace=0.2)

'''
  NEURAL ACTIVITY MEASURES FUNCTIONS
'''
## engagement-index activity functions
def engagement_index_vector(active_data,passive_data,t0,t1):
  active=[np.mean(a[:,t0:t1],0) for a in active_data]
  passive=[np.mean(p[:,t0:t1],0) for p in passive_data]

  return [(a-p)/np.linalg.norm(a-p) for a,p in zip(active,passive)]

def session_engagement_index(spks,ei_vector,t0=0,t1=250):
  return [np.dot(trl[t0:t1],ei_vector) for trl in spks]

def sessions_engagement_index(sessions_spks,sessions_ei_vectors,t0=0,t1=250):
  return [session_engagement_index(spks,ei_vector,t0,t1) \
    for spks,ei_vector in zip(sessions_spks,sessions_ei_vectors)]

def session_ei_plot(sessions_eis,index,ax):
  # print(index, sessions_eis['actv'][index])
    ax.scatter(list(range(len(sessions_eis['actv'][index]))),sessions_eis['actv'][index])
    ax.scatter(list(range(len(sessions_eis['psv'][index]))),sessions_eis['psv'][index])
    # sns.distplot(a=sessions_eis['actv'][index],kde=True,ax=ax)
    # sns.distplot(a=sessions_eis['psv'][index],kde=True,ax=ax)
    ax.legend(['active','passive'])
    ax.set_title(sessions_params[index]['mouse_name']+' '+sessions_params[index]['date_exp'])

def sessions_ei_plot(sessions_eis=[],columns_index=[],n_rows=0,n_cols=0,figure_size=()):
  fig,ax = plt.subplots(n_rows,n_cols,figsize=figure_size)
  for r in range(n_rows):
    for c in range(n_cols):
      index=(r*n_cols)+c
      if index==len(columns_index):
        return
      else:
        session_ei_plot(sessions_eis,columns_index[index],ax[r,c])

def region_engagement_index_analysis(region_data,t0=5,t1=25,plot=False):
  research_object=research()
  research_object.parameters['t0']=t0
  research_object.parameters['t1']=t1

  research_object.parameters['region_spks']=region_data
  research_object.parameters['ei_vectors']=engagement_index_vector(region_data['actv'],region_data['psv'],t0,t1)
  research_object.parameters['eis']={
    'actv':sessions_engagement_index(region_data['actv'],research_object.parameters['ei_vectors'],t0,t1),
    'psv':sessions_engagement_index(region_data['psv'],research_object.parameters['ei_vectors'],t0,t1)
    }

  _length,useful_sessions_index=check_valid_sessions(research_object.parameters['eis']['actv'])
  while _length%2!=0:
    _length+=1
  if _length%3==0:
    nrows=3
  elif _length%2==0:
    nrows=2
  ncols=int(_length/nrows)
  research_object.parameters['plot_params']={
    'sessions_eis':research_object.parameters['eis'],
    'columns_index':useful_sessions_index,
    'n_rows':nrows,
    'n_cols':ncols,
    'figure_size':(20,10)
  }
  research_object.methods['plot']=sessions_ei_plot

  return research_object

'''
    Granger methods
'''
def stationarity_test(X):
  return adfuller(X)[1] < 0.05

def granger_causality(X,Y,lag):
  if stationarity_test(X) == 0 or stationarity_test(Y) == 0:
    X=np.diff(X)[1:]
    Y=np.diff(Y)[1:]
  data=pd.DataFrame(np.vstack((X,Y)).T, columns=['Y','X'])
  result = grangercausalitytests(data,maxlag=lag,verbose=False)
  p_value = [round(result[i+1][0]['ssr_chi2test'][1],4) for i in range(lag)]
  return min(p_value)

def bi_region_delays_granger_test(X,Y,delays,lag=1,t0=50,t1=250):
  t0 = int(t0/10)
  t1 = int(t1/10)
  delay_list=[d for d in delays]

  trials_p_values=list()

  for x,y in zip(X,Y):
    p_values=list()
    for td in delay_list:
        if ((x[t0:t1] == y[t0+td:t1+td]).all()==True) or np. all(x[t0:t1]==x[t0:t1][0]) or np.all(y[t0+td:t1+td]==y[t0+td:t1+td][0]):
            p_values.append(1)
        else:
            p_values.append(granger_causality(y[t0+td:t1+td],x[t0:t1],lag))
        trials_p_values.append(p_values)

  return pd.DataFrame(np.array(trials_p_values),columns=delay_list)

'''
  LCC methods
'''

def lcc_scores_delays_plot(dataframe=[],height=6,aspect=1.3):
  sns.regplot(data=dataframe, x='delays',y='lcc_scores')


def bi_region_delays_LCC_test(Xs,delays,t0=5,t1=25):
  research_object=research()
  delay_list=[d for d in delays]
  lcc_scores=list()

  for td in delay_list:
    XY=[ Xs[0][:,t0:t1], Xs[1][:,t0+td:t1+td] ]
    _lcc=LCC.LCC_param(XY)
    lcc_scores.append(_lcc._linearity)
  
  research_object.parameters['t0']=t0
  research_object.parameters['t1']=t1

  research_object.parameters['lcc_scores']=pd.DataFrame(
    {
      'delays':delay_list,
      'lcc_scores':lcc_scores
    }
  )

  research_object.parameters['plot_params']={
    'dataframe':research_object.parameters['lcc_scores']
  }
  research_object.methods['plot']=lcc_scores_delays_plot

  return research_object