import pandas as pd
import matplotlib
import stoch_corr
from stoch_corr import ModifiedOU as mou
import numpy as np

df = pd.read_csv("Pdata/commodity_close_prices", parse_dates = True, index_col = 0)

df_l_ret = np.log(df/df.shift(1))*100
rt = df_l_ret[['SLV', 'GLD']]
window_1 = 30
window_2 = 60
window_3 = 100
n_days = 1

sample = rt[1000:]

2001


i = mou()

scp_params = i.get_coeff(sample, window_1, 1)
#scp_params = i.compute_correlation_values(sample, 30)
print(scp_params)