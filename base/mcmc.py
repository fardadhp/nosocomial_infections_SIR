import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
from EpiModel import EpiEquations
from setup import importData
from setup import setupModel
from core import runModel 
from scipy.integrate import odeint
import theano.tensor as tt
import arviz as az
from LogLike import LogLike
import seaborn as sns


def errorFunction(theta, y0, tVector, data, sigma, allParams):
    allParams['generalConstantsValues'] = theta
    numberUnits = allParams['initialConditionsValues'].shape[1]
    y = runModel(allParams, y0, tVector)
    estC = y.iloc[-1,3*numberUnits:4*numberUnits].values
    estI = y.iloc[-1,4*numberUnits:5*numberUnits].values
    errC = (np.ma.log(data[0]).filled(0) - np.ma.log(estC).filled(0))
    errI = (np.ma.log(data[1]).filled(0) - np.ma.log(estI).filled(0))
    err = np.array([errC,errI])
    return -(0.5/sigma**2)*np.sum(err**2)

def main():    
    allParams = importData()
    y0, tVector = setupModel(allParams)
    numberUnits = allParams['initialConditionsValues'].shape[1]    
    sigma = 1
    sensitivity = pd.read_csv('./calibration/MCMC_priors.csv')
    calParams = pd.read_csv('./calibration/to_be_calibrated.csv', header=None).iloc[:,0].values
    for i in range(sensitivity.shape[0]):
        if not sensitivity.loc[i,'parameter'] in calParams:
            ind = np.where(allParams['generalParamsNames']==sensitivity.loc[i,'parameter'])[0][0]
            v = allParams['generalParamsValues'][ind]
            sensitivity.loc[i,'min'] = v
            sensitivity.loc[i,'max'] = v+0.00001
            
    avgC = np.array([np.mean(i) for i in allParams['fitData']['C_fitData']])
    avgI = np.array([np.mean(i) for i in allParams['fitData']['I_fitData']])
    obs = [avgC, avgI]
    # create our Op
    loglikeFunc = LogLike(errorFunction, y0, tVector, obs, sigma, allParams)
    runMCMC(loglikeFunc, sensitivity)
    plotCalibratedModel()

def runMCMC(loglikeFunc, sensitivity):    
    nSamples = 2000
    with pm.Model() as Model:
        theta = []
        for i in range(sensitivity.shape[0]):
            p = sensitivity.loc[i,'parameter']
            low = sensitivity.loc[i,'min']
            upp = sensitivity.loc[i,'max']
            theta.append(pm.Uniform(p, lower=low, upper=upp))
        theta = tt.as_tensor_variable(theta)
        pm.Potential('likelihood', loglikeFunc(theta))
        trace = pm.sample(nSamples, tune=int(nSamples/2), step=pm.Metropolis(), 
                          cores=8, chains=1)
        data = az.from_pymc3(trace=trace)
        az.plot_trace(trace)
        plt.savefig('./calibration/MCMC/traces.png',dpi=300)
        az.plot_posterior(trace)
        plt.savefig('./calibration/MCMC/posterior.png',dpi=300)
        modelSummary = az.summary(trace, round_to=4)
        modelSummary.to_csv('./calibration/MCMC/modelSummary.csv')
        y = pm.trace_to_dataframe(trace)
        y.to_csv('./calibration/MCMC/posterior_samples.csv', index=False)

def plotCalibratedModel():
    allParams = importData()
    numberUnits = allParams['initialConditionsValues'].shape[1]
    y0, tVector = setupModel(allParams)
    calParams = pd.read_csv('./calibration/MCMC/posterior_samples.csv')
    dataC = [np.mean(i) for i in allParams['fitData']['C_fitData']]
    dataI = [np.mean(i) for i in allParams['fitData']['I_fitData']]
    C,I = [],[]
    for i in range(calParams.shape[0]): 
        allParams['generalParamsValues'] = calParams.iloc[i,:].values
        y = runModel(allParams, y0, tVector)
        C.append(y.iloc[-1,3*numberUnits:4*numberUnits].values)
        I.append(y.iloc[-1,4*numberUnits:5*numberUnits].values)    
    ncol = 2
    fig, ax = plt.subplots(figsize=(16,(numberUnits+1)*8),nrows=numberUnits,ncols=ncol)
    for i in range(numberUnits):
        ci = [item[i] for item in C]
        ii = [item[i] for item in I]
        sns.boxplot(data=ci, ax=ax[i][0], color='#7fc97f')
        sns.boxplot(data=ii, ax=ax[i][1], color='#beaed4')
        ax[i][0].hlines(dataC[i], -0.375, 0.375, colors='red', linewidth=3)
        ax[i][1].hlines(dataI[i], -0.375, 0.375, colors='red', linewidth=3)
        ax[i][0].set_xticklabels(['detected colonized'])
        ax[i][0].set_ylabel('Unit '+str(i+1), fontsize=20)
        ax[i][1].set_xticklabels(['infected'])
    for i in range(numberUnits):
        for j in range(2):
            ax[i][j].tick_params(labelsize=20)
    fig.savefig("./calibration/MCMC/results.png", dpi=300)
    plt.close('all')


if __name__ == '__main__':
    main()
    
  
