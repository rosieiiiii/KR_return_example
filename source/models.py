import numpy as np


def one_fit(Z, # dim: (nt, Nmax-date_s)
            l_unscaled,
            rx, # (Bc_shift-B)/B - rf
            date_s, # number of day shifts
            K # 2D, specific to alpha and delta. no row/col for inf maturity
            ):
    '''
    - Z has shape (nt, Nmax-date_s), rx has shape (nt,)
    - K is 2D and specific to alpha and delta
    - B, C, B_shift should be scaled s.t. B=1
    '''

    Nmax=K.shape[0]
    nt=rx.shape[0]

    # get column indexes with nonzero cashflow
    arr_msk_col=np.where(Z.sum(axis=0)!=0)[0]
    # max ttm in days at t+1
    tau_max_inday=arr_msk_col[-1]+1
    # scale the ridge penalty term
    l_scaled = nt*l_unscaled/tau_max_inday

    K_masked=K.take(arr_msk_col,axis=0).take(arr_msk_col,axis=1)
    Z_masked=Z[:,arr_msk_col]

    # full model: get fit as rx_g_full
    # 0-th entry is rx of (s+1)-period security at t 
    # 1-st entry is rx of (s+2)-period security at t 
    rx_g_full=K.take(arr_msk_col,axis=1)@Z_masked.T\
        @np.linalg.inv(Z_masked@K_masked@Z_masked.T + l_scaled * np.eye(nt))@rx
    # goodness of fit based on fitting of rx
    rx_full=Z_masked@rx_g_full.take(arr_msk_col)
    rmse=np.sqrt(np.mean((rx-rx_full)**2))
    
    # alignment:
    # 0-th entry is rx of 1-period security at t
    # s-st entry is rx of s+1-period security at t. The first non-nan entry
    # (N-1)-st entry is rx of N-period security at t
    rx_g_full_aligned = np.roll(rx_g_full,date_s)
    rx_g_full_aligned[:date_s]=np.nan

    # results from the full model
    dict_full={'rx_g_full':rx_g_full, # rx_g fitted by full model
               'rx_g_full_aligned':rx_g_full_aligned, # rx_g aligned 
               'rx_full':rx_full, # rx of underlying securities fitted by full model
               'rmse':rmse # based on fitting of rx, not rx_g
              }

    return dict_full


def FM_ridge_solution(Z, # dim: (nt, Nmax-date_s)
            l_unscaled,
            rx, # (Bc_shift-B)/B - rf
            date_s, # number of day shifts
            V, # svd of K
            D_diag, # svd of K
            R, # max number of factors, fit for r=1,2,...,R
            fit_single_R=False, # if true, only fit for r=R. for use in CV
            lst_R_fit=None # list of r used for fitting
            ):


    Nmax=V.shape[0]
    nt=rx.shape[0]

    # get column indexes with nonzero cashflow
    arr_msk_col=np.where(Z.sum(axis=0)!=0)[0]
    # max ttm in days at t+1
    tau_max_inday=arr_msk_col[-1]+1
    # scale the ridge penalty term
    l_scaled = nt*l_unscaled/tau_max_inday

    def FM_fit_r(r):
        V_fm=V[:,:r]
        D_diag_fm=D_diag[:r]

        beta_g_fm=V_fm@np.diag(np.sqrt(D_diag_fm)) # N by r
        beta_fm=Z.take(arr_msk_col, axis=1)@beta_g_fm.take(arr_msk_col, axis=0) # nt by r

        w_fm=beta_fm.T@np.linalg.inv(beta_fm@beta_fm.T + l_scaled*np.eye(nt)) # r by nt
        F_rx_fm=w_fm@rx # shape: r,
        rx_fm=beta_fm@F_rx_fm # fitted rx. shape: nt
        # goodness of fit based on fitting of rx with FM
        rmse_fm=np.sqrt(np.mean((rx-rx_fm)**2))
        # reconstructed rx_g
        # 0-th entry is rx of (s+1)-period security at t 
        # 1-st entry is rx of (s+2)-period security at t 
        rx_g_fm=beta_g_fm@F_rx_fm
        
        # alignment:
        # 0-th entry is rx of 1-period security at t
        # 1-st entry is rx of 2-period security at t
        # (s-1)-st entry is rx of s-period security at t
        # s-st entry is rx of s+1-period security at t. The first non-nan entry
        # (N-1)-st entry is rx of N-period security at t
        rx_g_fm_aligned = np.roll(rx_g_fm,date_s)
        rx_g_fm_aligned[:date_s]=np.nan

        dict_fm_r={'rx_g_fm':rx_g_fm,
                   'rx_g_fm_aligned':rx_g_fm_aligned,
                    'rx_fm':rx_fm,
                    'rmse_fm':rmse_fm,
                    'w_fm':w_fm,
                    'F_rx_fm':F_rx_fm
                   }
        return dict_fm_r


    if fit_single_R:
        return FM_fit_r(R)
    elif lst_R_fit is not None:
        dict_fm={}
        for r in lst_R_fit:
            dict_fm[r]=FM_fit_r(r)
        return dict_fm
    else:
        dict_fm={}
        for r in range(1,R+1):
            dict_fm[r]=FM_fit_r(r)
        return dict_fm


def leave_one_out_cv_fixed_alpha(Z, # dim: (nt, Nmax-date_s)
            lst_l_in,
            rx, # (Bc_shift-B)/B - rf
            date_s, # number of day shifts
            lst_train_ind,
            lst_test_ind,
            # specific to full model
            K=None, # 2D, specific to alpha
            # specific to factor model
           r=None,
           V=None,
           D_diag=None,
           # options for this function
           model_type='KR_full'):
    '''
    - Z has shape (nt, Nmax-date_s), rx has shape (nt,)
    - K_Nmax is 2D and specific to alpha
    - return arr_rmse_oos with shape (num_l,)
    '''

    #print('len lst_l_in: {}'.format(len(lst_l_in)))
    nt=rx.shape[0]
    num_fold=len(lst_train_ind)
    assert model_type in ['KR_full','KR_FM_ridge']
    if model_type in ['KR_FM_ridge']: 
        assert (r is not None and \
            V is not None and \
            D_diag is not None)
        Nmax=V.shape[0]
    else:
        assert (K is not None)
        Nmax=K.shape[0]

    #average oos pricing error for each lambda
    arr_rmse_oos=np.zeros(len(lst_l_in)) #(num_l,) 
    arr_rmse_is=np.zeros(len(lst_l_in)) #(num_l,) in-sample
    for k in range(num_fold):
        ind_fit,ind_ts=lst_train_ind[k],lst_test_ind[k]
        Z_fit=Z[ind_fit,:]     
        Z_ts=Z[ind_ts,:] 
        arr_rx_g_solved=np.full((len(lst_l_in),Nmax),np.nan)

        for i,l_unscaled in enumerate(lst_l_in):
            if model_type=='KR_full':
                dict_out=one_fit(Z=Z_fit,
                                l_unscaled=l_unscaled,
                                rx=rx[ind_fit], 
                                date_s=date_s, 
                                K=K) 
                arr_rmse_is[i]+=dict_out['rmse']
                arr_rx_g_solved[i,:]=dict_out['rx_g_full']

            elif model_type=='KR_FM_ridge':
                dict_out=FM_ridge_solution(Z=Z_fit,
                                l_unscaled=l_unscaled,
                                rx=rx[ind_fit], 
                                date_s=date_s,
                                V=V,
                                D_diag=D_diag,
                                R=r,
                                fit_single_R=True)

                arr_rmse_is[i]+=dict_out['rmse_fm'] #np.sqrt((dict_out['err_insample']**2).mean())
                arr_rx_g_solved[i,:]=dict_out['rx_g_fm']


        arr_rmse_oos+=(((Z_ts@arr_rx_g_solved[:,:Z_ts.shape[1]].T)-rx[ind_ts])**2).squeeze()
    arr_rmse_oos=np.sqrt(arr_rmse_oos/nt)
    arr_rmse_is=arr_rmse_is/num_fold

    rmse_oos_best=np.nanmin(arr_rmse_oos)
    l_best=lst_l_in[np.nanargmin(arr_rmse_oos)]

    dict_out={'arr_rmse_oos':arr_rmse_oos,
            'rmse_oos_best':rmse_oos_best,
            'l_best':l_best}
    return dict_out


def leave_one_out_cv(Z,
            lst_l_in,
            lst_alpha_in,
            rx, # (Bc_shift-B)/B - rf
            date_s, # number of day shifts
            K_Nmax_stack # 3D
            ):
    '''
    - This function is only for the full model only
    '''
    
    nt=rx.shape[0]
    lst_test_ind=[[i] for i in np.arange(nt)]
    lst_train_ind=[np.setdiff1d(np.arange(nt),test_ind) for test_ind in lst_test_ind]

    rmse_oos_best=np.inf
    alpha_best,l_best=None,None

    arr_rmse_oos=np.full((len(lst_alpha_in),len(lst_l_in)),np.nan)
    for i,alpha in enumerate(lst_alpha_in):
        dict_out=leave_one_out_cv_fixed_alpha(Z=Z,
                                lst_l_in=lst_l_in,
                                rx=rx,
                                date_s=date_s,
                                lst_train_ind=lst_train_ind,
                                lst_test_ind=lst_test_ind,
                                K=K_Nmax_stack[i,:,:], # 2D, specific to alpha
                               model_type='KR_full')
        rmse_oos_best_temp=dict_out['rmse_oos_best']
        l_best_temp=dict_out['l_best']

        arr_rmse_oos[i,:]=dict_out['arr_rmse_oos']

        if rmse_oos_best_temp<rmse_oos_best:
            rmse_oos_best=rmse_oos_best_temp
            alpha_best=alpha
            l_best=l_best_temp
            
    return l_best,alpha_best, arr_rmse_oos