import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from setup import importData
from setup import setupModel
from EpiModel import EpiEquations
from scipy.integrate import odeint


def runModel(allParams, y0, tVector):
    numberUnits = allParams['initialConditionsValues'].shape[1]
    funArgs = (numberUnits,*allParams['generalParamsValues'],allParams['unitParamsValues'],\
               allParams['admission'],allParams['discharge'],allParams['transferIn'],\
               allParams['transferOut'],allParams['internalTransferRate'],\
               allParams['deviceTransferRate'])
    y = odeint(EpiEquations, y0, tVector, args=tuple(funArgs))
    y = pd.DataFrame(y)
    return y

def plotResults(y, numberUnits):
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    lbls = ['S', 'X', 'UC', ' DC', 'I', 'M', 'N0', 'N1', 'D0', 'D1', 'Y', 'L', 'W']
    cols = []
    for l in lbls:
        for u in range(1,numberUnits+1):
            cols.extend([l+"_"+str(u)])
    nc = 4
    for u in range(numberUnits):
        fig, ax = plt.subplots(figsize=(16,16),nrows=math.ceil(len(lbls)/nc),ncols=nc)
        for i in range(len(lbls)):
            ax[i//nc][i%nc].plot(y.iloc[:,i*numberUnits+u])
            ax[i//nc][i%nc].set_title(lbls[i],fontsize=16)
            ax[i//nc][i%nc].tick_params(labelsize=12)
        for j in range(i+1, ax.shape[0]*ax.shape[1]):
            ax[j//nc][j%nc].set_visible(False)
        # legendValues = ['unit_'+str(u) for u in np.arange(1,numberUnits+1)]
        # lgd = fig.legend(legendValues, ncol=2, fontsize=16, loc=8)
        fig.tight_layout()
        fig.savefig("./results/Unit_"+str(u+1)+".png", dpi=300) #bbox_extra_artists=(lgd,)
        plt.close(fig)
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

