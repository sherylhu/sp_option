# Aurthor: Sherry Hu
import numpy as np
import pandas as pd

# Import data series from excel
datafile = "spread_option.xlsx"
xslx = pd.ExcelFile(datafile)
sheetName = "DCCGARCH"
dataSeries = xslx.parse(sheetName)
# ============= # NAGARCH(1 ,1) #============
# set asset 1
fData = dataSeries[ "asset1"].values
fInitParams = [0.2, 0.5, 0.0008, 0.75]
fParams = maximizeMLE(fInitParams, fData)
fGarchVol = GARCHLeveraged(fParams, fData)
fStdRet = fData / np.sqrt(fGarchVol)

sData = dataSeries["asset2"].values
sInitParams = [0.2, 0.5, 0.0008, 0.75]
sParams = maximizeMLE(sInitParams, sData)
sGarchVol = GARCHLeveraged(sParams, sData)
sStdRet = sData / np.sqrt(sGarchVol)


# ====== # Lamda estimation #====
def computeQ(lda):
    qf = np.ones(np.size(fData))
    qfs = np.ones(np.size(fData))
    qs = np.ones(np.size(fData))

    q = pd.DataFrame(data=np.array([qf, qfs, qs]).T,
                 columns=["f-f", "f-s", "s-s"], index = dataSeries.index )
q["f-f"][0] = 1
q["s-s"][0] = (fStdRet * sStdRet).sum() / np.size(fStdRet)
q["s-s"][0] = 1

for t in range(1, np.size(fData)):
    q["f-f"][t]= (1-lda)*fStdRet[t-1]**2 + lda*q["f-s"][t-1]
    q["f-s"][t]= (1-lda)*fStdRet[t-1]*sStdRet[t-1] + lda*q["f-s"][t-1]
    q["s-s"][t]= (1-lda)*sStdRet[t-1]**2 + lda*q["f-f"][t-1]

return q


def dccMinLikelihood(lda):
    q = computeQ(lda)
    r = q["f-s"] / np.sqrt(q["f-f"] * q["s-s"])
    logL = -1 / 2*((np.log(1-r**2)) + (fStdRet*2+sStdRet**2-2*r*fStdRet*sStdRet) / (1-r**2))
    return -(logL.sum())

bnds = [(0.00001, 0.99999)]
initParam = 0.94
resultDcc = optimize.minimize(dccMinLikelihood, initParam,
method ='SLSQP', bounds = bnds)
print(resultDcc)


def maximizeMLE(initParams , data ):
# Constraints
# 1 - (alpha*(1 + thetaˆ2)+beta) >= 0
    def persistenceIndexConstraint (params ):
        omega , alpha , beta = params
        return 1 - ( alpha + beta )
    cons = {'type' : 'ineq', 'fun' : persistenceIndexConstraint} # Could be also a lambda expression
# cons = {’type’ : ’ineq’, ’fun’ : lambda params : 1 - np.sum(params)}
# Bounds
# 0 <= parameters <= 1
    bnds = ((0, 1), (0, 1), (0, 1))
# Run the minimizer
    results = optimize.minimize(minLikelihood, initParams, data, method='SLSQP', bounds=bnds, constraints=cons)
    return results .x

def GARCHLeveraged(params , data ):
    alpha , beta , omega , theta = params
    s = np.zeros(np.size(data)) 
    s[0] = np.var(data, ddof=1)
    
    for i in range(1, np.size(data)):
        s[i] = omega + alpha*((data[i-1] - theta * np.sqrt(s[i-1]))**2) + beta*(s[i-1])
    return s