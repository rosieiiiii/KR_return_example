import numpy as np
import time 
import os
import argparse
import pickle
import pandas as pd
import scipy.optimize as opt #for ytm fitting


def get_ytm_and_duration(cashflow,time_to_cashflow_inday, B_i, y_guess=0.0):
    '''
    - Calculate annualized YTM (not in %) and duration (in years) of a security
    - YTM is estimated using Newton's method
    - Assume (1) continuous compounding and (2) each year has 365 days
    - Args:
        - cashflow (numpy array): amount of cashflow
        - time_to_cashflow_inday (numpy array): time to cashflow in days
        - B_i (float): price of the security
        - y_guess (float): initial guess for YTM in Newton's method
    - Returns:
        - ytm_solved (float): estimated YTM
        - dur_solved (float): estimated duration in years
    '''

    assert time_to_cashflow_inday.shape==cashflow.shape
    ytm_func=lambda y: (sum(cashflow*np.exp(-time_to_cashflow_inday/365*y))-B_i)**2

    ytm_solved=opt.newton(ytm_func,y_guess)
    dur_solved=sum((time_to_cashflow_inday/365)*cashflow*np.exp(-time_to_cashflow_inday/365*ytm_solved))/B_i

    return ytm_solved, dur_solved
    


def print_matrix(A,lst_row_header=None,num_digit=4,lst_col_header=None):
    '''
    - to use: print(print_matrix(A))
    - lst_row_header is header for row, lst_col_header is header for column
    '''
    str_out=''
    (num_row,num_col)=A.shape
    if lst_row_header is not None:
        assert len(lst_row_header)==num_row
    if lst_col_header is not None:
        for i,header in enumerate(lst_col_header):
            if i==len(lst_col_header)-1:
                str_out+=header+'\\\\'
            else:
                str_out+=header+'&'
        str_out+='\n'
        
    for i in range(0,num_row):
        str_temp=''
        if lst_row_header is not None:
            str_temp+=lst_row_header[i]+'&'
        for j in range(0,num_col):
            str_temp+=np.array2string(A[i,j],precision=num_digit)
            if j==num_col-1:
                str_temp+='\\\\'
            else:
                str_temp+='&'
        #str_temp=str_temp.replace('<','$<$').replace('>','$>$')
        str_out+=str_temp+'\n'
    return str_out

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        # raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
        # print(f"readable_dir:{path} is not a valid path")
        return path
