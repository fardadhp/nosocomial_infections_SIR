import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def importData(timeStep):
    allParams = {}
    # unit names
    unitNames = pd.read_csv("./data/unit data/names.csv",header=None).iloc[:,0].values
    allParams['unitNames'] = unitNames
    # admissiom/discharge/transfer
    admission = []
    discharge = []
    transferIn = []
    transferOut = []
    for unit in unitNames:
        file = './data/unit data/'+unit+'.csv'
        unitData = pd.read_csv(file).loc[:,['admission','discharge','transfer_in','transfer_out']]
        admission.append(unitData.loc[:,'admission'].values)
        discharge.append(unitData.loc[:,'discharge'].values)
        transferIn.append(unitData.loc[:,'transfer_in'].values)
        transferOut.append(unitData.loc[:,'transfer_out'].values)
    allParams['admissionTS'] = admission
    allParams['dischargeTS'] = discharge
    allParams['transferInTS'] = transferIn
    allParams['transferOutTS'] = transferOut
    # general Parameters
    generalParams = pd.read_csv("./data/general_parameters.csv",header=None)
    generalParamsNames = generalParams.iloc[:,0].values
    generalParamsValues = generalParams.iloc[:,1].values
    allParams['generalParamsNames'] = generalParamsNames
    allParams['generalParamsValues'] = generalParamsValues
    # unit Parameters
    unitParams = pd.read_csv("./data/unit data/unit_parameters.csv", index_col=0)
    unitParamsNames = np.array(unitParams.index)
    unitParamsValues = unitParams.loc[:, unitNames].values
    allParams['unitParamsNames'] = unitParamsNames
    allParams['unitParamsValues'] = unitParamsValues
    # internal patient transfer
    internalTransferData = pd.read_csv("./data/patient_transfer_internal.csv")
    allParams['internalTransferData'] = internalTransferData
    # device transfer
    deviceTransferRate = pd.read_csv("./data/device_transfer_rate.csv",index_col=[0])
    allParams['deviceTransferRate'] = deviceTransferRate.loc[unitNames, unitNames].values
    # initial conditions
    initialConditions = pd.read_csv("./data/initial_conditions.csv", index_col=0)
    initialConditionsParams = np.array(initialConditions.index)
    initialConditionsValues = initialConditions.loc[:, unitNames].values
    allParams['initialConditionsParams'] = initialConditionsParams
    allParams['initialConditionsValues'] = initialConditionsValues
    # patient status
    hyperParams = pd.read_csv('./data/patient_status_hyperparameters.csv')
    allParams['patientStatusHyperParams'] = hyperParams
    # fitting data
    S_fitData = []
    X_fitData = []
    C_fitData = []
    I_fitData = []
    for unit in unitNames:
        file = './data/unit data/'+unit+'.csv'
        fitData = pd.read_csv(file)
        S_fitData.append(fitData.loc[:,'Susceptible'].values)
        X_fitData.append(fitData.loc[:,'SusceptibleX'].values)
        C_fitData.append(fitData.loc[:,'Colonized'].values)
        I_fitData.append(fitData.loc[:,'Infected'].values)
    allParams['fitData'] = {'S_fitData': S_fitData, 'X_fitData': X_fitData, 'C_fitData': C_fitData, 'I_fitData': I_fitData}    
    allParams = adjustToTimeStep(allParams, timeStep)
    return allParams

def adjustToTimeStep(allParams, timeStep):
    unitlessConstants = pd.read_csv('./data/list_of_unitless_constants.csv', header=None).iloc[:,0].values
    # general Parameters
    ind = [i for i, e in enumerate(allParams['generalParamsNames']) if e not in unitlessConstants]
    allParams['generalParamsValues'][ind] = [min(1,allParams['generalParamsValues'][i]*timeStep) for i in ind]
    # unit Parameters
    ind = [i for i, e in enumerate(allParams['unitParamsNames']) if e not in ['r_n', 'r_d']]
    allParams['unitParamsValues'][ind] = [allParams['unitParamsValues'][i]*timeStep for i in ind]
    # device transfer rate
    allParams['deviceTransferRate'] *= timeStep
    return allParams
    
    
def setupModel(allParams, timeStep, simLength=None):
    if type(simLength) == type(None):
        simLength = len(allParams['fitData']['I_fitData'][0])
    T = simLength * int(24/timeStep)
    S0, X0, UC0, DC0, I0, M0, Y0, L0, W0 = allParams['initialConditionsValues']
    P = S0 + X0 + UC0 + DC0 + I0
    N0_0 = P * allParams['unitParamsValues'][np.where(allParams['unitParamsNames']=='r_n')][0]
    N1_0 = np.zeros(len(P))
    D0_0 = P * allParams['unitParamsValues'][np.where(allParams['unitParamsNames']=='r_d')][0]
    D1_0 = np.zeros(len(P))
    y0 = np.hstack([S0, X0, UC0, DC0, I0, M0, N0_0, N1_0, D0_0, D1_0, Y0, L0, W0])
    tVector = np.linspace(0, T, T+1)
    return (y0, tVector)
    
    
    
    
    
    
    
