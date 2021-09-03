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
from core import runModel, patientFlow

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
    
    
def main(nSamples, timeStep, simLength):
    # import Params and data
    allParams = importData(timeStep)
    numberUnits = allParams['initialConditionsValues'].shape[1]
    # setup model
    y0, tVector = setupModel(allParams, timeStep, simLength)
    # run
    y = runModel(allParams, y0, tVector, timeStep, simLength)
    # plot
    plotResults(y, numberUnits)

if __name__ == '__main__':
    try:
        nSamples, timeStep = [int(i) for i in sys.argv[1:]]
    except:
        nSamples = 4
        timeStep = 1  # hour(s)
        simLength = 100
    main(nSamples, timeStep, simLength)

