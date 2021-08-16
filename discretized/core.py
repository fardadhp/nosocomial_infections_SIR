import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from setup import importData
from setup import setupModel
from EpiModel import EpiEquations
from scipy.integrate import odeint

def patientFlow(allParams, t):
    numberUnits = allParams['initialConditionsValues'].shape[1]
    if t % 24 == 0:
        tt = int(t//24)
        admission = np.array([unit[tt] for unit in allParams['admissionTS']])
        discharge = np.array([unit[tt] for unit in allParams['dischargeTS']])
        extTransfer = np.array([unit[tt] for unit in allParams['transferTS']])
    else:
        admission, discharge, extTransfer = [0,0,0]
    allParams['admission'] = admission / 24
    allParams['discharge'] = discharge / 24
    extTransferIn = np.clip(extTransfer, a_min=0, a_max=np.inf) / 24
    extTransferOut = np.clip(-extTransfer, a_min=0, a_max=np.inf) / 24
    allParams['transferIn'] = extTransferIn
    allParams['transferOut'] = extTransferOut 
    
    return allParams
        

def runModel(allParams, y0, tVector):
    numberUnits = allParams['initialConditionsValues'].shape[1]    
    y = []
    for i in range(int(len(tVector)/24)):
        tVec = tVector[i*24:(i+1)*24]
        allParams = patientFlow(allParams, tVec[0])
        funArgs = (numberUnits,*allParams['generalParamsValues'],allParams['unitParamsValues'],\
               allParams['admission'],allParams['discharge'],allParams['transferIn'],\
               allParams['transferOut'],allParams['internalTransferRate'],\
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

