import pandas as pd
import numpy as np
from datetime import datetime
import scipy.sparse as sps
import pickle
import time 
import argparse
import multiprocessing as mp
import os
import sys

import utils
import kernel
import models


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--freq', type=str, default='daily')
parser.add_argument("--flg_mp", type=utils.str2bool, nargs='?',
                        const=True, default=False,
                        help="use multiprocessing")
parser.add_argument('--num_t_each_trunk', type=int, default=30)
parser.add_argument('--max_num_process', type=int, default=3)


parser.add_argument('--R', type=int, default=3, 
    help='(>0): max number of factors for factor modle')
parser.add_argument('--lst_R_fit', type=str, default='',
     help='list for r by ",". Will override other options for R.' )
parser.add_argument("--use_maturity_mask", type=utils.str2bool, nargs='?',
                        const=True, default=False,
                        help="whether to use the 90-day maturity mask for filtering input")


parser.add_argument('--l_fixed', type=float, default=1,
     help='ridge')
parser.add_argument('--alpha_fixed', type=float, default=0.05,
     help='the fixed value for alpha')
parser.add_argument('--delta_fixed', type=float, default=0.0,
     help='the fixed value for delta.')


parser.add_argument('--dir_out_base', type=utils.dir_path, default='./',help='where to save output')
parser.add_argument('--idx_ver', type=int, default=1, 
    help='index of version')

args = parser.parse_args()
assert args.freq=='daily'

if args.lst_R_fit:
    lst_R_fit=[int(r) for r in args.lst_R_fit.split(',')]
else:
    lst_R_fit=None

### where to save output
base_name='ret_curve_R_{}_alpha_{}_ridge_{}_RF_{}_{}_ver_{}'\
    .format(args.R, args.alpha_fixed, args.l_fixed, args.rf_source, args.freq, args.idx_ver)

dir_out=args.dir_out_base+base_name+'/'
if not os.path.isdir(dir_out):
    os.makedirs(dir_out)
print('*'*3 +'Save results to: '+ '*'*3+'\n'+dir_out)

### get path and load lookup table
dir_B = './Data/B_max_ttm_10yr/'
df_t_lookup_freq=pd.read_pickle(dir_B+'df_t_lookup_{}.pkl'.format(args.freq))
df_t_lookup_daily=pd.read_pickle(dir_B+'df_t_lookup_{}.pkl'.format('daily'))

B_mat=np.load(dir_B+'B_mat.npy')
Bc_shift_mat=np.load(dir_B+'Bc_shift_mat.npy')
with open(dir_B+"dict_par.pkl", "rb") as handle:
    dict_par = pickle.load(handle)
nmax,Nmax,prefix_C,dir_npz=[dict_par[key] for key in ['nmax','Nmax','prefix_C','npz_dir']]


### load daily risk-free rate
df_rf=pd.read_pickle('./data_supplement/df_riskfree_daily_all.pkl').KR_LS


### load mask
dir_mask='./mask/'
mat_mask_maturity=np.load(dir_mask+'mat_filter_maturity_90days.npy')
mat_nt=np.load(dir_mask+'mat_nt.npy')
mat_ytm=np.load(dir_mask+'mat_ytm.npy')


### load daily discount curve
df_g_daily = pd.read_pickle('./data_supplement/df_kr_g.pkl')


### generate kernel matrix
K = kernel.generate_kernel_matrix(args.alpha_fixed, args.delta_fixed, Nmax, Nmax)
# SVD
U,D_diag,Vh = np.linalg.svd(K)
V = Vh.T
D = np.diag(D_diag) # D is a matrix
DV_inv = V@np.diag(1/np.sqrt(D_diag))
# assert np.isclose(U[:,:10],V[:,:10]).all()
dict_svd = {'V':V,
            'D_diag':D_diag,
            'DV_inv':DV_inv}



def main():  
    #save parameters
    dict_discount_curve_par=dict()
    dict_discount_curve_par.update(dict_par)
    if args.l_fixed==-1:
         dict_discount_curve_par['lst_l']=lst_l
    dict_discount_curve_par.update(vars(args))

    with open(dir_out+'dict_ret_curve_par.pkl','wb') as handle:
        pickle.dump(dict_discount_curve_par,handle,protocol=pickle.HIGHEST_PROTOCOL)  
    
    start_time=time.time()
    if args.flg_mp:
        mp.set_start_method("spawn")
        pool = mp.Pool(processes=args.max_num_process)
        
        lst_t_sample=np.arange(0,len(df_t_lookup_freq) ,1) # set last arg to 1 if estimate for every month
        
        num_trunk=int(np.ceil(len(lst_t_sample)/args.num_t_each_trunk))

        for i in range(0,num_trunk):
            lst_ind=np.arange(i*args.num_t_each_trunk,np.min([(i+1)*args.num_t_each_trunk,len(lst_t_sample)]))
            lst_t=lst_t_sample[lst_ind]
            pool.apply_async(mp_discount_curve_solution, args=(lst_t,))
        pool.close()
        pool.join() 
    else:
        lst_t_sample=np.arange(0,len(df_t_lookup_freq) ,1) # set last arg to 1 if estimate for every month
        
        mp_discount_curve_solution(lst_t_sample)
    print('exiting main: execution time: {:.2f} minutes'.format((time.time() - start_time)/60))

def mp_discount_curve_solution(lst_t_freq):


    for t_freq in lst_t_freq:
        t=df_t_lookup_freq.iloc[t_freq].t
        today=df_t_lookup_freq.index[t_freq]
        today_str=today.strftime('%Y-%m-%d')

        # return over x days. find the subsequent date
        if t+1==len(df_t_lookup_daily) or np.count_nonzero(B_mat[:,t])==0: 
            # can't find the subsequent date or no eligible securities today
            print('skipping t:{}, t_freq:{}, date:{}'.format(t,t_freq,today_str))
            continue

        date_s=(df_t_lookup_daily.index[t+1]-df_t_lookup_daily.index[t]).days
        B=B_mat[:,t]
        nt=int(mat_nt[t])
        
        # filter step
        if args.use_maturity_mask:
            mask_keep=mat_mask_maturity[t,:nt]
            mask_keep=np.logical_and(mask_keep, mat_ytm[t,:nt] < 0.25) 
        else:
            mask_keep=np.full(nt, True)

        csr_mat_name=dir_npz+prefix_C+'C_'+str(t)+'.npz'
        csr_mat=sps.load_npz(csr_mat_name)

        # remove empty rows from B and C
        # apply filter
        B=B[:nt][mask_keep]
        Bc_shift=Bc_shift_mat[:nt,t][mask_keep]
        C=csr_mat.toarray()[:nt,1:][mask_keep]
        nt=len(B)

        print('t:{}, t_freq:{}, date:{}, nt: {}, date_s:{}'.format(t,t_freq,today_str, nt, date_s))

        # normalize prices to 1
        Bc_shift=(1/B)*Bc_shift
        C=(1/B)[:,np.newaxis]*C
        B=np.ones(nt)*1

        ### the next cash flow should be on t+date_s. No cashflow in between by construction
        # assert C[:,:date_s-1].sum()==0

        # get return of securities
        rf=(1+df_rf.loc[today])**date_s-1 # scalar
        ret=(Bc_shift-B)/B
        rx=ret-rf

        # get one-day excess return of zcb
        g=df_g_daily.iloc[t].values
        # g_shift=df_g_daily.iloc[t+1].values
        # g_shift_2=np.roll(g_shift,date_s)
        # g_shift_2[:date_s]=1
        # rx_g=(g_shift_2-g)/g-rf # the first (date_s-1) values are artificial and should be discarded

        #C_tilde=C@np.diag(g[:Nmax])

        Z_bar=C[:,date_s-1:]@np.diag(g[date_s-1:Nmax]) # dim: (nt, Nmax-date_s+1)
        Z=Z_bar[:,1:] # dim: (nt, Nmax-date_s)

        dict_full=models.one_fit(Z=Z,
            l_unscaled=args.l_fixed,
            rx=rx, # (Bc_shift-B)/B - rf
            date_s=date_s, # number of day shifts
            K=K # 2D, specific to alpha
            )

        # factor model
        dict_fm=models.FM_ridge_solution(Z=Z,
            l_unscaled=args.l_fixed,
            rx=rx, # (Bc_shift-B)/B - rf
            date_s=date_s, # number of day shifts
            V=dict_svd['V'], # svd of K
            D_diag=dict_svd['D_diag'], # svd of K
            R=args.R, # max number of factors
            lst_R_fit=lst_R_fit # list of r used for fitting. will override R above
            )
        
        # compile output
        dict_out={
              'rx': rx, # rx of underlying securities
              'ret':ret, # ret of underlying securities
              'date_s':date_s, # date spacing
              'rf':rf, # compounded rf with date_s
              'dict_full':dict_full,
              'dict_fm':dict_fm
             }

        with open(dir_out+'dict_ret_curve_tfreq_{}_t_{}.pkl'.format(t_freq,t),'wb') as handle:
            pickle.dump(dict_out,handle,protocol=pickle.HIGHEST_PROTOCOL)   

if __name__=='__main__':
    main()












