import mgarch
import numpy as np
from collections import deque

class DCC:
    def get_cond_corr(self, df, n_days, dist = 'norm'):  #rolling conditional covariance matrix
        '''
        MGARCH model used to fit data to extract the conditional covariance matrix
        Returns 1-day ahead predictions

        TO DO: EXTEND THIS BEYOND THE 2 VARIABLE CASE for a M variable case and extract the given correlation matrix, flatten it into a list and append to a dataframe for later analysis. 
        This will be useful within contagion events
        '''
        _corr = df.corr()
        m_len = df.shape[1]
        _sigma = deque(maxlen = m_len)  #empty sigma list to append values to
        _vol = mgarch.mgarch(dist)
        _vol.fit(df)
        _x = _vol.predict(n_days)
        _cvm = np.array(_x.get('cov'))
        sigs = np.diag(_cvm)

        for i in range(m_len):
            s = sigs[i]
            _sigma.append(np.sqrt(s))  #append std_dev

        _L = np.linalg.cholesky(_cvm)
        _corr = _L[1][0]/_sigma[1]
        return {'L':_L, 'CVM': _cvm, 'cond_rho':_corr, 'r_cor' : _corr}
    
    def conditional_corr(self, data, window, n_days, dist):
        cond_corr_series = deque(maxlen = len(data) - window)

        for i in range(len(data) - window):
            rt = data[i:window+i]
            x = self.get_cond_corr(rt, n_days, dist)
            cond_corr_series.append( (x.get('cond_rho'), x.get('r_corr') )  )

        return cond_corr_series