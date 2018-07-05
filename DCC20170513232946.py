# -*- coding: utf-8 -*-
"""
@author: dtoppo

Dynamic Conditional Correlations

1) Compute the unconditional sample covariance and correlation for the equity index return series of Germany and Japan.

2) Calculate the unconditional 1-month, 1% Value-at-Risk for a portfolio consisting of 50% invested in each market. Calculate also the 1-month, 1% Value-at-Risk for each asset individually. Use the normal distribution. Compare the portfolio VaR with the sum of individual VaRs. What can you note? Why are they not the same?

3) Estimate a separate NAGARCH(1,1) model for each of the equity return series. Standardize each return using the implied (filtered) NAGARCH standard deviation.

4) Use QMLE to estimate the exponential smoother version of the dynamic conditional correlation (DCC) model for the two equity markets. Set the starting value of λ at 0.94. 

5) Estimate a GARCH(1,1)-t(d) model using MLE (jointly estimate all parameters d, α, β, ω and Q). Please use Italian stock index returns.


Data: data_python.xlsx
Tab: DCC
"""

import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize

from GARCHLeverage import GARCHLeveraged, maximizeMLE

# Import dataserie from Excel
dataFile = "data_python.xlsx"
xslx = pd.ExcelFile(dataFile)

sheetName = "DCC"

dataSeries = xslx.parse(sheetName)


#==============================================================================
# Unconditional sample covariance and correlation
#==============================================================================
sampleCov = np.cov(dataSeries[["Spain","Sweden"]].T, ddof=0)[0,1]
sampleCorr = np.corrcoef(dataSeries[["Spain","Sweden"]].T, ddof=0)[0,1]

#==============================================================================
# Portfolio unconditional covariance
#==============================================================================
spainWgt = 0.5;
swedenWgt = 0.5;

spainVar = np.var(dataSeries["Spain"], ddof=1)
swedenVar = np.var(dataSeries["Sweden"], ddof=1)

# PPF = percent point function === quantile function = norminv
# q = prob(X<=x)
VaRSpain = -sp.stats.norm.ppf(q=0.01, loc=0, scale=1)*np.sqrt(spainVar)*spainWgt
VaRSweden = -sp.stats.norm.ppf(q=0.01, loc=0, scale=1)*np.sqrt(swedenVar)*swedenWgt
sumOfVaRs = VaRSpain + VaRSweden

pfCov = np.cov(dataSeries[["Spain","Sweden"]].T, ddof=0)[0,1]
pfVar = spainWgt**2*spainVar + swedenWgt**2*swedenVar + 2*spainWgt*swedenWgt*pfCov

VaRPf = -sp.stats.norm.ppf(q=0.01, loc=0, scale=1)*np.sqrt(pfVar)

#==============================================================================
# NAGARCH(1,1)
#==============================================================================
spData = dataSeries["Spain"].values

spInitParams = [0.2, 0.5, 0.0008, 0.75]
spParams = maximizeMLE(spInitParams, spData)
spGarchVol = GARCHLeveraged(spParams, spData)
spStdRet = spData / np.sqrt(spGarchVol)


swData = dataSeries["Sweden"].values

swInitParams = [0.2, 0.5, 0.0008, 0.75]
swParams = maximizeMLE(swInitParams, swData)
swGarchVol = GARCHLeveraged(swParams, swData)
swStdRet = swData / np.sqrt(swGarchVol)

#==============================================================================
# Lamda estimation
#==============================================================================
def computeQ(lda):
    qDE = np.ones(np.size(spData))
    qDEJP = np.ones(np.size(spData))
    qJP = np.ones(np.size(spData))

    q = pd.DataFrame(data=np.array([qDE, qDEJP, qJP]).T, columns=["SP-SP", "SP-SW", "SW-SW"], index=dataSeries.index)
    
    q["SP-SP"][0] = 1
    q["SP-SW"][0] = (spStdRet * swStdRet).sum() / np.size(spStdRet)
    q["SW-SW"][0] = 1
    
    for t in range(1, np.size(spData)):
        q["SP-SP"][t] = (1-lda)*spStdRet[t]**2 + lda*q["SP-SP"][t-1]
        q["SP-SW"][t] = (1-lda)*spStdRet[t]*swStdRet[t] + lda*q["SP-SW"][t-1]
        q["SW-SW"][t] = (1-lda)*swStdRet[t]**2 + lda*q["SW-SW"][t-1]
        
    return q

def dccMinLikelihood(lda):
    q = computeQ(lda)
    r = q["SP-SW"] / np.sqrt(q["SP-SP"] * q["SW-SW"])
    logL = -1/2*((np.log(1-r**2)) + (spStdRet**2+swStdRet**2-2*r*spStdRet*swStdRet)/(1-r**2))
    return -(logL.sum())

bnds = [(0, 1)]

initParam = 0.94

resultDcc = optimize.minimize(dccMinLikelihood, initParam, method='SLSQP', bounds=bnds)

print(resultDcc)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        