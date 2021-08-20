import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from setup import importData
from setup import setupModel
from EpiModel import EpiEquations
from scipy.integrate import odeint
import scipy.stats as stats

def patientFlow(allParams, t, timeStep):
    numberUnits = allParams['initialConditionsValues'].shape[1]
    if t % int(24/timeStep) == 0:
        tt = int(t//int(24/timeStep))
        admission = np.array([unit[tt] for unit in allParams['admissionTS']])
        discharge = np.array([unit[tt] for unit in allParams['dischargeTS']])
        extTransferIn = np.array([unit[tt] for unit in allParams['transferInTS']])
        extTransferOut = np.array([unit[tt] for unit in allParams['transferOutTS']])
    else:
        admission, discharge, extTransferIn, extTransferOut = [0,0,0,0]
    allParams['admission'] = admission / int(24/timeStep)
    allParams['discharge'] = discharge / int(24/timeStep)
    # extTransferIn = np.clip(extTransfer, a_min=0, a_max=np.inf) / int(24/timeStep)
    # extTransferOut = np.clip(-extTransfer, a_min=0, a_max=np.inf) / int(24/timeStep)
    allParams['transferIn'] = extTransferIn / int(24/timeStep)
    allParams['transferOut'] = extTransferOut / int(24/timeStep)    
    return allParams

def samplePatientStatus(allParams):
    ind = np.where(allParams['generalParamsNames']=='muAdmC')[0][0]
    muAdmC = allParams['generalParamsValues'][ind]
    sigmaAdmC = allParams['generalParamsValues'][ind+1]
    muAdmI = allParams['generalParamsValues'][ind+2]
    sigmaAdmI = allParams['generalParamsValues'][ind+3]
    muTranC = allParams['generalParamsValues'][ind+4]
    sigmaTranC = allParams['generalParamsValues'][ind+5]
    muTranI = allParams['generalParamsValues'][ind+6]
    sigmaTranI = allParams['generalParamsValues'][ind+7]
    admC = stats.truncnorm((-muAdmC)/sigmaAdmC, (1 - muAdmC)/sigmaAdmC, loc=muAdmC, scale=sigmaAdmC).rvs(1)[0]
    admI = stats.truncnorm((-muAdmI)/sigmaAdmI, (1 - admC - muAdmI)/sigmaAdmI, loc=muAdmI, scale=sigmaAdmI).rvs(1)[0]
    tranC = stats.truncnorm((-muTranC)/sigmaTranC, (1 - muTranC)/sigmaTranC, loc=muTranC, scale=sigmaTranC).rvs(1)[0]
    tranI = stats.truncnorm((-muTranI)/sigmaTranI, (1 - tranC - muTranI)/sigmaTranI, loc=muTranI, scale=sigmaTranI).rvs(1)[0]
    return (admC, admI, 1, tranC, tranI, 1)
        
        

def runModel(allParams, y0, tVector, timeStep, simLength):
    numberUnits = allParams['initialConditionsValues'].shape[1]    
    y = []
    ind = np.where(allParams['generalParamsNames']=='muAdmC')[0][0]
    for i in range(simLength):
        tVec = tVector[i*int(24/timeStep):(i+1)*int(24/timeStep)+1]
        patientStatus = samplePatientStatus(allParams)
        allParams = patientFlow(allParams, tVec[0], timeStep)
        funArgs = (numberUnits,*allParams['generalParamsValues'][:ind],*patientStatus,\
                   allParams['unitParamsValues'],allParams['admission'],allParams['discharge'],\
                   allParams['transferIn'],allParams['transferOut'],allParams['internalTransferRate'],\
                   allParams['deviceTransferRate'])
        output = odeint(EpiEquations, y0, tVec, args=funArgs)
        if any(output[-1]<-1):
            raise ValueError('negative value in compartments!')
        y.append(output[-1])
        y0 = output[-1]  
    y = pd.DataFrame(y)
    return y

def plotResults(y, numberUnits):
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    lbls = ['S', 'X', 'UC', ' DC', 'I', 'M', 'N0', 'N1', 'D0', 'D1', 'Y', 'L', 'W']
    cols = []
    for l in lbls:
        for u in range(1,numberUnits+1):
            cols.extend([l+"_"+str(u)])
    nc = 4
    for u in range(numberUnits):
        fig1, ax1 = plt.subplots(figsize=(16,16),nrows=math.ceil(len(lbls)/nc),ncols=nc)
        fig2, ax2 = plt.subplots(figsize=(16,16),nrows=math.ceil(len(lbls)/nc),ncols=nc)
        for i in range(len(lbls)):
            ax1[i//nc][i%nc].plot(y.iloc[:,i*numberUnits+u])
            ax1[i//nc][i%nc].set_title(lbls[i],fontsize=16)
            ax1[i//nc][i%nc].tick_params(labelsize=12)
            ax2[i//nc][i%nc].plot(np.round(y.iloc[:,i*numberUnits+u]))
            ax2[i//nc][i%nc].set_title(lbls[i],fontsize=16)
            ax2[i//nc][i%nc].tick_params(labelsize=12)
        for j in range(i+1, ax1.shape[0]*ax1.shape[1]):
            ax1[j//nc][j%nc].set_visible(False)
            ax2[j//nc][j%nc].set_visible(False)
        fig1.tight_layout()
        fig2.tight_layout()
        fig1.savefig("./results/Unit_"+str(u+1)+".png", dpi=300)
        fig2.savefig("./results/Unit_"+str(u+1)+"_rounded.png", dpi=300)
        plt.close('all')
    y.columns = cols
    y.to_csv("./results/compartments.csv", index=False)

def plotFitData(allParams):
    numberUnits = allParams['initialConditionsValues'].shape[1]
    fig, ax = plt.subplots(figsize=(16,16),nrows=4,ncols=2)
    i = 0
    for k,v in allParams['fitData'].items():
        for u in range(numberUnits):
            ax[i][u].plot(pd.DataFrame(np.transpose(v)).iloc[:,u])
            ax[i][u].set_title(k,fontsize=16)
            ax[i][u].tick_params(labelsize=12)
        i += 1
    fig.tight_layout()
    fig.savefig("./results/fit_data.png",dpi=300)    

