import os
import copy
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from datetime import datetime
from pandas.tseries.offsets import *
import argparse

import utils


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--username', type=str, required=True)
parser.add_argument("--download_data_wrds", type=utils.str2bool, nargs='?',
                        const=True, default=True,
                        help="download data from WRDS")
args = parser.parse_args()


### Download raw data from WRDS
dir_data = './wrds_raw_data/' # where to save and read WRDS raw data
dir_output = './processed_data/' # where to save processed data output

if not os.path.exists(dir_data):
    os.makedirs(dir_data)
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

if args.download_data_wrds:
    import wrds
    db = wrds.Connection(wrds_username=args.username)

    print('Downloading TFZ_ISS ...')
    df_iss = db.get_table(library='crsp', table='TFZ_ISS')
    df_iss.to_pickle(dir_data+'df_iss.pkl')
    
    print('Downloading TFZ_MAST ...')
    df_mast = db.get_table(library='crsp', table='TFZ_MAST')
    df_mast.to_pickle(dir_data+'df_mast.pkl')
    
    print('Downloading TFZ_PAY ...')
    df_pay = db.get_table(library='crsp', table='TFZ_PAY')
    df_pay.to_pickle(dir_data+'df_pay.pkl')
    
    print('Downloading TFZ_DLY ...')
    df_dly = db.get_table(library='crsp', table='TFZ_DLY')
    df_dly.to_pickle(dir_data+'df_dly.pkl')
    
    db.close()
else:
    print('Loading raw data ...')
    df_iss = pd.read_pickle(dir_data+'df_iss.pkl')
    df_mast = pd.read_pickle(dir_data+'df_mast.pkl')
    df_pay = pd.read_pickle(dir_data+'df_pay.pkl')
    df_dly = pd.read_pickle(dir_data+'df_dly.pkl')
    

### Format raw data
# change all column names to upper case
for df in [df_iss, df_mast, df_pay, df_dly]:
    df.columns = [col.upper() for col in df.columns]
    
df_pay['KYCRSPID'] = df_pay['KYCRSPID'].astype(object)
df_dly['KYCRSPID'] = df_dly['KYCRSPID'].astype(object)
df_dly['TDIDXRATIO_FLG'] = df_dly['TDIDXRATIO_FLG'].astype(object)

# merge df_iss and df_mast into df_info, containing bond information
df_info = df_iss.join(df_mast.set_index('KYTREASNO'),on='KYTREASNO',how='outer')

# convert all date into pandas datetime format
for col in ['TDATDT','TMATDT','TFCPDT','TFCALDT','TBANKDT','TMFSTDAT',
            'TMLSTDAT','TDFSTDAT','TDLSTDAT']:
    df_info[col] = pd.to_datetime(df_info[col],format='%Y-%m-%d')
    
# calculate rounded maturity, rounded to the nearest year. This is for filtering later on.
df_info['RoundedMaturityYears'] = np.round(((df_info['TMATDT']-df_info['TDATDT']).copy()\
                                          /np.timedelta64(1, 'Y')).values)


### Format daily prices
#replace missing values with NaN
df_dly.TDRETNUA.replace(-99, np.nan, inplace = True)
df_dly.TDYLD.replace(-99, np.nan, inplace = True)
df_dly.TDDURATN.replace(-1, np.nan, inplace = True)
df_dly.TDBID.replace(0, np.nan, inplace = True)
df_dly.TDASK.replace(0, np.nan, inplace = True)
df_dly.TDNOMPRC.replace(0, np.nan, inplace = True)

#convert quotation date CALDT into datetime format
df_dly['CALDT'] = pd.to_datetime(df_dly['CALDT'],format='%Y-%m-%d')

# slice df_info where KYTREASNO is in df_dly
# i.e. retain information of only securities whose prices are available
df_info_dly = df_info[df_info.KYTREASNO.isin(df_dly.KYTREASNO.unique())]


### Format coupon payment information
df_pay['TPQDATE'] = pd.to_datetime(df_pay['TPQDATE'],format='%Y-%m-%d')


### Get nominal price
dly_caldt = df_dly.CALDT.unique()
dly_caldt.sort()
dly_kytreasno = df_dly.KYTREASNO.unique()
T = len(dly_caldt)
N = len(dly_kytreasno)

df_nomprc = pd.DataFrame(np.nan,index=dly_caldt,columns=dly_kytreasno)
#fill df_nomprc with nominal prices from df_dly
for kytreasno in dly_kytreasno:
    temp_caldt = df_dly.loc[df_dly.KYTREASNO==kytreasno].CALDT
    temp_tdnomprc = df_dly.loc[df_dly.KYTREASNO==kytreasno].TDNOMPRC
    df_nomprc.loc[temp_caldt.values,kytreasno] = temp_tdnomprc.values
    
    
### Get daily unadjusted returns
df_tdretnua = pd.DataFrame(np.nan, index=dly_caldt, columns=dly_kytreasno)

for kytreasno in dly_kytreasno:
    temp_caldt = df_dly.loc[df_dly.KYTREASNO==kytreasno].CALDT
    temp_tdretnua = df_dly.loc[df_dly.KYTREASNO==kytreasno].TDRETNUA
    
    df_tdretnua.loc[temp_caldt.values,kytreasno] = temp_tdretnua.values
    
### Select data
# bond that are still quotes 0 (used to be np.nan)
print('number of bonds that are still quotes: {}'.format((df_info_dly.IWHY==0).sum()))
#bonds that have matured (iwhy==1)
print('number of bonds that have matured: {}'.format((df_info_dly.IWHY==1).sum()))
#bonds that are called for redemption (iwhy==2)
print('number of bonds that are called: {}'.format((df_info_dly.IWHY==2).sum()))
#bonds that are all exchanged (iwhy==3)
print('number of bonds that are all exchanged: {}'.format((df_info_dly.IWHY==3).sum()))
#Sources no longer quote issue (iwhy==4)
print('Sources no longer quote issue: {}'.format((df_info_dly.IWHY==4).sum()))

# keep only: nonflower, taxable bonds, bonds whose time series end because of maturity.
# remove certificate of deposit (ITYPE=3)
df_info_dly_s = df_info_dly[(df_info_dly.ITYPE.isin([1,2,4]))&\
(df_info_dly.ITAX==1)&\
(df_info_dly.IFLWR==1)&\
(df_info_dly.IWHY!=3)]

df_dly_s = df_dly[df_dly.KYTREASNO.isin(df_info_dly_s.KYTREASNO)]
df_pay_s = df_pay[df_pay.KYTREASNO.isin(df_info_dly_s.KYTREASNO)]

dly_caldt = df_dly_s.CALDT.unique()
dly_caldt.sort()
dly_kytreasno = df_dly_s.KYTREASNO.unique()
df_nomprc_s = pd.DataFrame(np.nan,index=dly_caldt,columns=dly_kytreasno)
df_tdretnua_s = pd.DataFrame(np.nan,index=dly_caldt,columns=dly_kytreasno)
df_tdaccint_s = pd.DataFrame(np.nan,index=dly_caldt,columns=dly_kytreasno)
df_tdpdint_s = pd.DataFrame(np.nan,index=dly_caldt,columns=dly_kytreasno)

#fill df_nomprc and df_tdretnua with nominal prices from df_dly_s
for kytreasno in dly_kytreasno:
    temp_caldt = df_dly_s.loc[df_dly_s.KYTREASNO==kytreasno].CALDT
    temp_tdnomprc = df_dly_s.loc[df_dly_s.KYTREASNO==kytreasno].TDNOMPRC
    temp_tdretnua=df_dly_s.loc[df_dly_s.KYTREASNO==kytreasno].TDRETNUA
    temp_tdaccint = df_dly_s.loc[df_dly_s.KYTREASNO==kytreasno].TDACCINT
    temp_tdpdint = df_dly_s.loc[df_dly_s.KYTREASNO==kytreasno].TDPDINT
    
    df_nomprc_s.loc[temp_caldt.values,kytreasno] = temp_tdnomprc.values
    df_tdretnua_s.loc[temp_caldt.values,kytreasno] = temp_tdretnua.values
    df_tdaccint_s.loc[temp_caldt.values,kytreasno] = temp_tdaccint.values
    df_tdpdint_s.loc[temp_caldt.values,kytreasno] = temp_tdpdint.values 
    
# B is ex-dividend
# Bc is cum dividend
df_dly_s['B'] = df_dly_s.TDNOMPRC + df_dly_s.TDACCINT
df_dly_s['Bc'] = df_dly_s.TDNOMPRC + df_dly_s.TDACCINT + df_dly_s.TDPDINT

df_B = pd.DataFrame(np.nan, index=dly_caldt, columns=dly_kytreasno)
df_Bc = pd.DataFrame(np.nan, index=dly_caldt, columns=dly_kytreasno)

#fill df_B and df_Bc 
for kytreasno in dly_kytreasno:
    temp_caldt = df_dly_s.loc[df_dly_s.KYTREASNO==kytreasno].CALDT
    temp_tdnomprc = df_dly_s.loc[df_dly_s.KYTREASNO==kytreasno][['B','Bc']]
    df_B.loc[temp_caldt.values,kytreasno] = temp_tdnomprc.B.values
    df_Bc.loc[temp_caldt.values,kytreasno] = temp_tdnomprc.Bc.values
    

### Save processed data
df_info_dly_s.to_pickle(dir_output+'df_info_dly_s.pkl')
df_dly_s.to_pickle(dir_output+'df_dly_s.pkl')
df_pay_s.to_pickle(dir_output+'df_pay_s.pkl')
df_nomprc_s.to_pickle(dir_output+'df_nomprc_s.pkl')
df_tdretnua_s.to_pickle(dir_output+'df_tdretnua_s.pkl')
df_tdaccint_s.to_pickle(dir_output+'df_tdaccint_s.pkl')
df_tdpdint_s.to_pickle(dir_output+'df_tdpdint_s.pkl')

df_dly_s.to_pickle(dir_output+'df_dly_B_s.pkl')
df_B.to_pickle(dir_output+'df_B_s.pkl')
df_Bc.to_pickle(dir_output+'df_Bc_s.pkl')
print('successfully saved')
