import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
from EpiModel import EpiEquations
from setup import importData
from setup import setupModel
from core import runModel, patientFlow, samplePatientStatus
from scipy.integrate import odeint
import theano.tensor as tt
import arviz as az
from LogLike import LogLike
import seaborn as sns
import os
import glob
import sys
import time
import multiprocessing as mp


def errorFunction(theta, y0, tVector, data, sigma, allParams, timeStep):
    allParams['generalParamsValues'] = theta
    numberUnits = allParams['initialConditionsValues'].shape[1]
    y = []
    ind = np.where(allParams['generalParamsNames']=='muAdmC')[0][0]
    for i in range(int(len(tVector)/int(24/timeStep))):
        tVec = tVector[i*int(24/timeStep):(i+1)*int(24/timeStep)]
        patientStatus = samplePatientStatus(allParams)
        allParams = patientFlow(allParams, tVec[0], timeStep)
        funArgs = (numberUnits,*allParams['generalParamsValues'][:ind],*patientStatus,allParams['unitParamsValues'],\
               allParams['admission'],allParams['discharge'],allParams['transferIn'],\
               allParams['transferOut'],allParams['intTransferRate'],\
               allParams['deviceTransferRate'])
        output = odeint(EpiEquations, y0, tVec, args=tuple(funArgs))
        if any(output[-1]<-1):
            raise ValueError('negative value in compartments!')
        y.append(output[-1])
        y0 = output[-1]
    y = pd.DataFrame(y)
    estC = [] 
    estI = []
    for j in range(numberUnits):
        estC.append(y.iloc[:,3*numberUnits+j].values)
        estI.append(y.iloc[:,4*numberUnits+j].values)
    estC = np.array(estC)
    estI = np.array(estI)
    errC = (np.ma.log(data[0]).filled(0) - np.ma.log(estC).filled(0))
    errI = (np.ma.log(data[1]).filled(0) - np.ma.log(estI).filled(0))
    err = np.array([errC,errI])
    return -(0.5/sigma**2)*np.sum(err**2)
    

def runMCMC(loglikeFunc, sensitivity, allParams, nSamples, nCore):    
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
                          cores=nCore, chains=min(4, nCore))
        data = az.from_pymc3(trace=trace)
        az.plot_trace(trace)
        plt.savefig('./calibration/MCMC/traces.png',dpi=300)
        az.plot_posterior(trace)
        plt.savefig('./calibration/MCMC/posterior.png',dpi=300)
        modelSummary = az.summary(trace, round_to=4)
        modelSummary.to_csv('./calibration/MCMC/modelSummary.csv')
        y = pm.trace_to_dataframe(trace)
        y.to_csv('./calibration/MCMC/posterior_samples.csv', index=False)

def plotCalibratedModel(timeStep, simLength):
    allParams = importData(timeStep)
    numberUnits = allParams['initialConditionsValues'].shape[1]
    y0, tVector = setupModel(allParams, timeStep, simLength)
    calParams = pd.read_csv('./calibration/MCMC/posterior_samples.csv')
    dataC = [item[:len(tVector)//int(24/timeStep)] for item in allParams['fitData']['C_fitData']]
    dataI = [item[:len(tVector)//int(24/timeStep)] for item in allParams['fitData']['I_fitData']]
    C,I = [],[]
    for i in range(calParams.shape[0]): 
        allParams['generalParamsValues'] = calParams.iloc[i,:].values
        y = runModel(allParams, y0, tVector, timeStep, simLength)
        C.append(y.iloc[:,3*numberUnits:4*numberUnits].values)
        I.append(y.iloc[:,4*numberUnits:5*numberUnits].values)
    minC = []
    meanC = []
    maxC = []
    minI = []
    meanI = []
    maxI = []
    for i in range(len(C[0])):
        minC.append(pd.DataFrame([it[i] for it in C]).min(0).values)
        meanC.append(pd.DataFrame([it[i] for it in C]).mean(0).values)
        maxC.append(pd.DataFrame([it[i] for it in C]).max(0).values)
        minI.append(pd.DataFrame([it[i] for it in I]).min(0).values)
        meanI.append(pd.DataFrame([it[i] for it in I]).mean(0).values)
        maxI.append(pd.DataFrame([it[i] for it in I]).max(0).values)
    
    ncol = 2
    fig, ax = plt.subplots(figsize=(16,(numberUnits)*8),nrows=numberUnits,ncols=ncol)
    for i in range(numberUnits):
        ax[i][0].plot(dataC[i], color='red', label='data')
        ax[i][0].plot(np.transpose(meanC)[i], color='black', linestyle='--', label='model')
        ax[i][0].fill_between(np.arange(simLength), np.transpose(minC)[i], np.transpose(meanC)[i], color='#7fc97f', label='95% CI')
        ax[i][0].fill_between(np.arange(simLength), np.transpose(maxC)[i], np.transpose(meanC)[i], color='#7fc97f')
        ax[i][1].plot(dataI[i], color='red', label='data')
        ax[i][1].plot(np.transpose(meanI)[i], color='black', linestyle='--', label='model')        
        ax[i][1].fill_between(np.arange(simLength), np.transpose(minI)[i], np.transpose(meanI)[i], color='#beaed4', label='95% CI')
        ax[i][1].fill_between(np.arange(simLength), np.transpose(maxI)[i], np.transpose(meanI)[i], color='#beaed4')
        ax[i][0].set_xticklabels(['detected colonized'])
        ax[i][0].set_ylabel('Unit '+str(i+1), fontsize=20)
        ax[i][1].set_xticklabels(['infected'])
        ax[i][0].legend()
        ax[i][1].legend()
    for i in range(numberUnits):
        for j in range(2):
            ax[i][j].tick_params(labelsize=20)
    fig.savefig("./calibration/MCMC/results.png", dpi=300)
    plt.close('all')

def main(nCore, nSamples, timeStep, simLength):
    if not os.path.isdir("./calibration/MCMC"):
        os.mkdir("./calibration/MCMC")
    allParams = importData(timeStep)
    y0, tVector = setupModel(allParams, timeStep, simLength)
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
            
    avgC = np.array([data[:int(len(tVector)//int(24/timeStep))] for data in allParams['fitData']['C_fitData']])
    avgI = np.array([data[:int(len(tVector)//int(24/timeStep))] for data in allParams['fitData']['I_fitData']])
    obs = [avgC, avgI]
    # create our Op
    loglikeFunc = LogLike(errorFunction, y0, tVector, obs, sigma, allParams, timeStep)
    runMCMC(loglikeFunc, sensitivity, allParams, nSamples, nCore)
    plotCalibratedModel(timeStep, simLength)
    
    
if __name__ == '__main__':
    try:
        nCore, nSamples, timeStep, simLength = [int(i) for i in sys.argv[1:]]
    except:
        nCore = mp.cpu_count()
        nSamples = 4
        timeStep = 1  # hour(s)
        simLength = 10
    t0 = time.time()    
    main(nCore, nSamples, timeStep, simLength)
    print(time.time()-t0)
    
  
