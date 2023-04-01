
import pandas as pd
from pandas import read_csv
import numpy as np
import sys
import os
import warnings
from matplotlib import pyplot as plt 
from pandas import read_csv
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import glob
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import *
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

if not sys.warnoptions:
    warnings.simplefilter("ignore")

cwd = os.path.abspath('')
file_list = os.listdir(cwd)
csv = glob.glob('*.{}'.format('csv'))

data_Main = pd.DataFrame(columns = ['DATE'])
for file in csv:
    temp = pd.read_csv(file, parse_dates= ['DATE'])
    data_Main = pd.DataFrame.merge(data_Main, temp, how = 'right', on = ['DATE'])


data_Main = data_Main.rename(columns = {'CES7000000008': 'L&H', 'WTISPLC': 'WTI', 
                                         'APU0000702111' : 'Bread', 'CUSR0000SAF116' : 'Alcohol', 'CSUSHPISA' : 'Home',
                                            'CPIAPPSL': 'Apparel'})
data_Main = data_Main.set_index('DATE')

maxlag=12
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def cointegration_test(df, alpha=0.05): 
    out = coint_johansen(df,-1,2)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Significance Level    = {signif}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")  

# granger = grangers_causation_matrix(data_Main, variables = ['L&H', 'WTI', 'Bread', 'Alcohol', 'Home', 'Apparel'])
# coint = cointegration_test(data_Main)
# print(granger)
# print(granger)
# print(data_Main)
data_Main = np.log(data_Main).diff().dropna()

# for name, column in data_Main.iteritems():
#     adfuller_test(column, name=column.name)
#     print('\n')

model = VAR(data_Main)
# for i in [1,2,3,4,5,6,7,8,9]:
#     result = model.fit(i)
#     print('Lag Order =', i)
#     print('AIC : ', result.aic)

model_fitted = model.fit(2)
model_irf = model_fitted.irf(5)
res = model_irf.irfs
cum_res = model_irf.cum_effects

for i in range(6):
    print(res[i][1])


model_irf.plot(response = 'L&H', impulse = 'WTI',
                       subplot_params = {'fontsize': 10}, seed = 1, signif = 0.05)

plt.show()

