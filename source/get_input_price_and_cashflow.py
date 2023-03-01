import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from IPython.display import display
from datetime import datetime
from pandas.tseries.offsets import *
%matplotlib inline
from itertools import groupby
import pickle

from tqdm.auto import tqdm
import os
import warnings
warnings.filterwarnings("ignore")


save_to_pickle = True
generate_C = True
mat_day = 3650 #time to maturity maturity cutoff

# where to save formatted data
os.chdir('..')
dir_output = './B_and_C/' 
# where to save B mat and date lookup tables
dir_B = dir_output+'B_max_ttm_10yr/'
# where to save C's
npz_dir = dir_output+'npz_C/'

if not os.path.exists(dir_output):
    os.mkdir(dir_output)
if not os.path.exists(dir_B):
    os.mkdir(dir_B)
if not os.path.exists(npz_dir):
    os.mkdir(npz_dir)

    
### Load selected data
dir_tfz = './processed_data/'
df_info_dly = pd.read_pickle(dir_tfz + 'df_info_dly_s.pkl')
df_dly = pd.read_pickle(dir_tfz + 'df_dly_s.pkl')
df_nomprc = pd.read_pickle(dir_tfz + 'df_nomprc_s.pkl')
df_tdaccint = pd.read_pickle(dir_tfz + 'df_tdaccint_s.pkl')
df_tdretnua = pd.read_pickle(dir_tfz + 'df_tdretnua_s.pkl')
df_tdpdint = pd.read_pickle(dir_tfz + 'df_tdpdint_s.pkl')
# shift date of daily return s.t. return from [t,t+1] is aligned at t
df_tdretnua_shift = df_tdretnua.shift(-1,axis=0)

df_pay = pd.read_pickle(dir_tfz + 'df_pay_s.pkl')
df_B = pd.read_pickle(dir_tfz + 'df_B_s.pkl')
df_Bc = pd.read_pickle(dir_tfz + 'df_Bc_s.pkl')

# shift date of Bc s.t. Bc at t+1 is aligned at t
df_Bc_shift = df_Bc.shift(-1, axis=0)
df_B_shift = df_B.shift(-1, axis=0)

df_nomprc_bin =~ df_nomprc.isnull()
df_tdretnua_bin =~ df_tdretnua_shift.isnull()

num_kytreasno = len(df_nomprc.columns)
nmax = df_nomprc_bin.sum(axis=1).max()

assert df_B.index.equals(df_nomprc.index)
assert (df_B.columns == df_nomprc.columns).all()
df_B_bin =~ df_B.isnull()
assert (df_B_bin == df_nomprc_bin).all().all()


### Get lookup table between t:0 to T-1 and dates
T = len(df_nomprc_bin.index)
# daily lookup
df_t_lookup = pd.DataFrame(index=df_nomprc.index,\
                         data=np.arange(0,T),columns=['t'])

display(df_t_lookup.head())

### monthly lookup
df_t_lookup['date'] = df_t_lookup.index
df_t_lookup_monthly = df_t_lookup\
    .groupby(by=[df_t_lookup.index.month, df_t_lookup.index.year]).max()\
    .reset_index()[['t','date']]\
    .set_index('date')\
    .sort_index()
df_t_lookup = df_t_lookup.drop(['date'], axis=1)


### Get dataframe of time to maturity
df_ttm = pd.DataFrame(index=df_nomprc.index)
pbar = tqdm(total=len(df_info_dly))

#fill df_ttm
for i in range(0, len(df_info_dly)):
    kytreasno = df_info_dly.iloc[i].KYTREASNO
    maturity_date = df_info_dly.iloc[i].TMATDT
    issue_date = df_info_dly.iloc[i].TDATDT
    time_to_maturity = (maturity_date-df_ttm.index).days
    #time_since_issue=(df_ttm.index-issue_date).days
    
    temp_ttm = (maturity_date-df_ttm.index).days.values.astype(np.float)
    # mark ttm of securities that have matured as 0
    temp_ttm[temp_ttm<0] = np.nan
    # mark ttm of securities that haven't been issued as 0
    temp_ttm[df_ttm.index<issue_date] = np.nan
    
    df_ttm[kytreasno] = temp_ttm#(maturity_date-df_ttm.index).days
    
    pbar.update(1)
    
### Generate cashflow matrix
print('mat_day cutoff:{}'.format(mat_day))
print('generate_C:{}'.format(generate_C))
print('T:{}'.format(T))
print('nmax:{}'.format(nmax))
print('num_kytreasno:{}'.format(num_kytreasno))
print('t=0 date:{}'.format(df_nomprc.index[0]))
print('t=T-1 date:{}'.format(df_nomprc.index[-1]))

prefix_C = 'C_10yr_'
Nmax_C = mat_day + 1 #the first col of C mat is 0 for convenience, will remove
removal_maturities = [2, 3, 4, 5, 7, 10, 20, 30]

# generate a dictionary of parameters and save it
dict_par = {'T':T, 'Nmax':mat_day, 'Nmax_C':Nmax_C, 'nmax':nmax,\
            't0':df_nomprc.index[0], 'tT-1':df_nomprc.index[-1],\
            'num_kytreasno':num_kytreasno, 'prefix_C':prefix_C, 'npz_dir':npz_dir}

# save date look-up table and dict_par
if save_to_pickle:
    df_t_lookup.to_pickle(dir_B + 'df_t_lookup_daily.pkl')
    df_t_lookup_monthly.to_pickle(dir_B + 'df_t_lookup_monthly.pkl')
    
    with open(dir_B + 'dict_par.pkl','wb') as handle:
        pickle.dump(dict_par,handle,protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('not saved')
    

    
        
    
print('Generating price vector and cashflows matrix ...')
#if save_to_pickle, WILL SAVE B_mat, ret_mat, ttm_day_mat AT THE END OF THIS BLOCK
B_mat = np.zeros([nmax,T])
Bc_shift_mat = np.zeros([nmax,T])
B_shift_mat = np.zeros([nmax,T])
ret_mat = np.full([nmax,T],np.nan) # aligned with B_mat
tdaccint_mat = np.full([nmax,T],np.nan) # aligned with B_mat
kytreasno_mat = np.full([nmax,T],np.nan) # aligned with B_mat
num_rm_on_the_run = np.zeros(T)


dict_B_kytreasno = dict() # track kytreasno used in B_mat and ret_mat
ttm_day_mat = np.full([nmax,T],np.nan) # track ttm in days, aligned with B_mat

pbar = tqdm(total=T)
for t in range(T):   
    #find kytreasno whose ttm is between (0, mat_day]
    df_ttm_slice = df_ttm.iloc[t]
    today = df_ttm_slice.name
    arr_kytreasno = df_ttm_slice[(df_ttm_slice>0)&(df_ttm_slice<=mat_day)].index.values

    #get B 
    srs_B = df_B.iloc[t][arr_kytreasno]
    srs_Bc_shift = df_Bc_shift.iloc[t][arr_kytreasno]
    srs_B_shift = df_B_shift.iloc[t][arr_kytreasno]
    #remove prices that are nan, this happen if bond has not been issued
    set_kytreasno_B = set(srs_B[~srs_B.isnull()].index)
    
    # get tdaccint
    srs_tdaccint = df_tdaccint.iloc[t][arr_kytreasno]

    # get return from t to t+1
    srs_ret = df_tdretnua_shift.iloc[t][arr_kytreasno]
    srs_ret[~srs_ret.isnull()].index
    set_kytreasno_ret = set(srs_ret[~srs_ret.isnull()].index)

    # list of kytreasno to use for time t
    lst_kytreasno = list(set_kytreasno_B.intersection(set_kytreasno_ret))
    
    
    ## Exclude the two most recently issued securities with 
    # maturities of 2, 3, 4, 5, 7, 10, 20, and 30 years for securities issued in 1980 or later.
    
    if today >= pd.to_datetime('1980-01-01'):
        remove_on_the_run = True
    else:
        remove_on_the_run = False

    if remove_on_the_run:
        df_info_slice = df_info_dly[df_info_dly.KYTREASNO.isin(lst_kytreasno)]
        lst_kytreasno_rm = []
        for maturity in removal_maturities:
            df_temp = df_info_slice[df_info_slice.RoundedMaturityYears==maturity]
            lst_kytreasno_rm.extend(list(df_temp.sort_values(by='TDATDT',ascending=False).iloc[:2].KYTREASNO.values))
        num_rm = len(lst_kytreasno_rm)
        lst_kytreasno = list(set(lst_kytreasno).difference(set(lst_kytreasno_rm)))
    else:
        num_rm = 0
    num_rm_on_the_run[t] = num_rm



    srs_B = srs_B.loc[lst_kytreasno]
    srs_Bc_shift = srs_Bc_shift.loc[lst_kytreasno]
    srs_B_shift = srs_B_shift.loc[lst_kytreasno]
    srs_ret = srs_ret.loc[lst_kytreasno]
    srs_tdaccint = srs_tdaccint.loc[lst_kytreasno]

    assert (srs_B.index==srs_ret.index).all()
    assert (srs_B.index==srs_Bc_shift.index).all()
    assert (srs_B.index==srs_B_shift.index).all()
    assert (srs_B.index==srs_tdaccint.index).all()
    num_prc=len(srs_B)

    #fill B_mat , ret_mat
    B_mat[0:num_prc, t] = srs_B.values
    Bc_shift_mat[0:num_prc, t] = srs_Bc_shift.values
    B_shift_mat[0:num_prc, t] = srs_B_shift.values
    ret_mat[0:num_prc, t] = srs_ret.values
    tdaccint_mat[0:num_prc, t] = srs_tdaccint.values
    kytreasno_mat[0:num_prc, t] = lst_kytreasno
    
    dict_B_kytreasno[t] = lst_kytreasno

    if generate_C:
        #fill C
        #get payment
        df_pay_valid_temp = df_pay[df_pay.KYTREASNO.isin(srs_B.index)]
        #assume a storage is given
        arr_C_temp = np.zeros([nmax,Nmax_C])
        # need to discard firsr col of arr_C_temp because no payment due today
        # where (timediff=0)   
        for i, kytreasno in enumerate(srs_B.index):
            #slice payment info corresponding to kytreasno
            df_pay_kytreasno_temp = df_pay_valid_temp\
            [df_pay_valid_temp.KYTREASNO==kytreasno]

            # calculate time to coupon payment as the time to ACTUAl payment 
            # i.e. (TDPDINT!=0), not according to scheduled payment date, which can be on weekend
            
            # get ACTUAL coupon payment dates
            df_slice = df_dly[df_dly.KYTREASNO==kytreasno][['CALDT','TDPDINT']]
            df_slice = df_slice[df_slice.TDPDINT!=0]
            # get the last coupon payment dates (i.e. maturity) when price quote ends
            if len(df_slice) > 0:
                df_slice_1 = df_pay_kytreasno_temp[df_pay_kytreasno_temp.TPQDATE<df_slice.CALDT.min()]
                df_slice_2 = df_pay_kytreasno_temp[df_pay_kytreasno_temp.TPQDATE>df_slice.CALDT.max()]
            else:
                df_slice_1 = None
                df_slice_2 = df_pay_kytreasno_temp[df_pay_kytreasno_temp.TPQDATE>=df_t_lookup.index[0]]
            df_slice_2 = df_slice_2[['TPQDATE','PDINT']]
            df_slice_2.columns = df_slice.columns
            # merge
            df_slice = pd.concat((df_slice,df_slice_2),ignore_index=True)
            # check no missing coupon payment dates
            if len(df_pay_kytreasno_temp[df_pay_kytreasno_temp.TPQDATE>=df_t_lookup.index[0]])!=len(df_slice):
                if len(df_slice_1) > 0:
                    df_slice_1 = df_slice_1[['TPQDATE','PDINT']]
                    df_slice_1.columns = df_slice.columns
                    df_slice = pd.concat((df_slice,df_slice_1),ignore_index=True)
                assert len(df_pay_kytreasno_temp[df_pay_kytreasno_temp.TPQDATE>=df_t_lookup.index[0]])==len(df_slice)
            df_slice.sort_values(by='CALDT', inplace=True)
            
            # fill C
            # calculate time left to coupon payment
            time_to_coupon_temp = df_slice.CALDT - today
            arr_day_to_coupon = time_to_coupon_temp.values.astype('timedelta64[D]').astype('int16')
            # add upcoming coupon payment to cashflow matrix
            # do not record cashflow today        
            arr_day_to_coupon_pos = arr_day_to_coupon[arr_day_to_coupon>0]
            arr_C_temp[i,arr_day_to_coupon_pos] = df_slice[arr_day_to_coupon>0].TDPDINT.values
            # time to maturity
            day_to_mat = (df_info_dly[df_info_dly.KYTREASNO==kytreasno].TMATDT-today).\
                values.astype('timedelta64[D]').astype('int16')
            # sanity check
            if len(arr_day_to_coupon) > 0: # coupon bond
                assert arr_day_to_coupon[-1]==day_to_mat
            ttm_day_mat[i,t] = day_to_mat
            #add face value payment
            arr_C_temp[i,day_to_mat] += 100     
            
        # the next cash flow should be on t+date_s. No cashflow in between by construction
        if t < T-1:
            date_s = (df_t_lookup.index[t+1]-df_t_lookup.index[t]).days
            assert arr_C_temp[:num_prc,1:][:,:date_s-1].sum()==0
            
        #convert to csr format and save to npz file
        csr_mat_temp = sps.csr_matrix(arr_C_temp)
        npz_filename = prefix_C + 'C_' + str(t) + '.npz'
        sps.save_npz(npz_dir + npz_filename, csr_mat_temp)
        
    pbar.update(1)
    

if save_to_pickle:
    np.save(dir_B + 'B_mat.npy', B_mat)
    np.save(dir_B + 'Bc_shift_mat.npy', Bc_shift_mat)
    np.save(dir_B + 'B_shift_mat.npy', B_shift_mat)
    np.save(dir_B + 'ret_mat.npy', ret_mat)
    np.save(dir_B + 'tdaccint_mat.npy', tdaccint_mat)
    np.save(dir_B + 'kytreasno_mat.npy', kytreasno_mat)
    np.save(dir_B + 'ttm_day_mat.npy', ttm_day_mat)
    with open(dir_B + 'dict_B_kytreasno.pkl', 'wb') as handle:
        pickle.dump(dict_B_kytreasno, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    print('not saved')      