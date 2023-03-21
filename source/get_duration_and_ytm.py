import os
import sys

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.optimize as opt 
from tqdm.auto import tqdm

import utils

dir_B = './B_and_C/B_max_ttm_10yr/'
df_t_lookup_daily = pd.read_pickle(dir_B + 'df_t_lookup_daily.pkl')
T = len(df_t_lookup_daily)

# read price vector 
B_mat = np.load(dir_B+'B_mat.npy')
with open(dir_B+"dict_par.pkl", "rb") as handle:
    dict_par = pickle.load(handle)
    
prefix_C,Nmax,nmax,dir_npz = [dict_par[key] for key in ['prefix_C','Nmax','nmax','npz_dir']]
arr_h = np.arange(1,Nmax+1)

dir_out = './mask/'

### get time to maturity in day
print('getting time to maturity in day ...')
mat_ttm = np.full((T-1,nmax),np.nan)
mat_nt = np.full(T-1,np.nan)

pbar = tqdm(total=T-1)
for t in range(T-1): 

    B = B_mat[:,t]
    csr_mat_name = dir_npz+prefix_C+'C_'+str(t)+'.npz'

    csr_mat = sps.load_npz(csr_mat_name)
    nt = np.count_nonzero(B)

    #remove empty rows from B and C
    B = B[:nt]
    C = csr_mat.toarray()[:nt,1:]
    #ttm in day, -1 to convert into index
    lst_ttm_inday = np.apply_along_axis(lambda row: len(row)-(row!=0)[::-1].argmax(),1,C)
    mat_ttm[t,:nt] = lst_ttm_inday
    mat_nt[t] = nt
    
    pbar.update(1)
    
np.save(dir_out + 'mat_ttm.npy', mat_ttm)
np.save(dir_out + 'mat_nt.npy', mat_nt)

### get maturity filter
print('getting maturity filter ...')
mat_filter = np.full((T-1, nmax), False)
for t in range(T-1):
    nt = int(mat_nt[t])
    arr_ttm = mat_ttm[t,:nt]
    mat_filter[t,:nt][arr_ttm>=90] = True
np.save(dir_out + 'mat_filter_maturity_90days.npy', mat_filter)

### Calculate YTM
print('Calculating YTM ...')
mat_dur = np.full((T-1,nmax),np.nan)
mat_ytm = np.full((T-1,nmax),np.nan)

pbar = tqdm(total=T-1)
for t in range(T-1):

    B = B_mat[:,t]
    csr_mat_name = dir_npz+prefix_C+'C_'+str(t)+'.npz'
    csr_mat = sps.load_npz(csr_mat_name)
    
    nt = int(mat_nt[t])
    B = B[:nt]
    C = csr_mat.toarray()[:nt,1:]

    for i in range(nt):
        C_i,B_i = C[i,:],B[i]
        t_cashflow = np.nonzero(C_i)[0]+1 # unit in days. time TO cashflow.
        cashflow = C_i[t_cashflow-1]
        #calculate annualized ytm and duration
        mat_ytm[t,i], mat_dur[t,i] = utils.get_ytm_and_duration(cashflow,t_cashflow,B_i)

    pbar.update(1)

np.save(dir_out + 'mat_dur.npy', mat_dur)
np.save(dir_out + 'mat_ytm.npy', mat_ytm)
print('saving complete')

