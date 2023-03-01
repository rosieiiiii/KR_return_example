import pandas as pd
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_in', type=str, default='../yield_panel_daily_frequency_daily_maturity.csv')
args = parser.parse_args()

y_KR = pd.read_csv(args.dir_in, index_col = 0)
y_KR.drop(columns=['MAX_DATA_TTM'], inplace=True)

Nmax = y_KR.shape[1]
g_KR = np.exp(-y_KR*np.arange(1,Nmax+1)/365)
df_kr = (1/g_KR['1']-1).to_frame(name='KR_LS')

g_KR.to_pickle('../data_supplement/df_kr_g.pkl')
df_kr.to_pickle('../data_supplement/df_riskfree_daily_all.pkl')