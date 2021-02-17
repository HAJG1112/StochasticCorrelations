import mgarch
import numpy as np
from collections import deque
import pandas as pd
from numba import jit
class DCC:
    
    def get_cond_corr(self, df, n_days, dist = 'norm'):  #rolling conditional covariance matrix
        '''
        MGARCH model used to fit data to extract the conditional covariance matrix
        Returns 1-day ahead predictions

        TO DO: add a Jit compiler to this function to speed it up
        '''
        m_len = df.shape[1]
        _sigma = []  #empty sigma list to append values to
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
        return {'L':_L, 'CVM': _cvm, 'cond_rho':_corr}
    
    def conditional_corr(self, data, window, n_days, dist):
        '''
        data: 2*N array consisting of the conditional correlation and realised over a window
        window: window size for correlation estimations
        n_days: days forecasted ahead (must be kept at 1 for now)
        dist: distribution used in the GARCH model. For now this is limited to the normal
              and student's-T distribution.
        '''
        cond_corr_series = deque(maxlen = len(data) - window)

        for i in range(len(data) - window):
            rt = data[i:window+i]
            _corr = rt.corr().iloc[0,1]
            x = self.get_cond_corr(rt, n_days, dist)
            cond_corr_series.append( (x.get('cond_rho'), _corr, (x.get('cond_rho') - _corr) ) ) 

        return cond_corr_series