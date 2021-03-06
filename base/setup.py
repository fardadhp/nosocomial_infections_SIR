import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


def importData():
    allParams = {}
    # admission, discharge, external transfer
    patientFlow = pd.read_csv("./data/external_patient_flow.csv", index_col=0)
    admission = patientFlow.loc['psi_in',:].values
    discharge = patientFlow.loc['psi_out',:].values
    transferIn = patientFlow.loc['n_in',:].values
    transferOut = patientFlow.loc['n_out',:].values
    allParams['admission'] = admission
    allParams['discharge'] = discharge
    allParams['transferIn'] = transferIn
    allParams['transferOut'] = transferOut
    # general Parameters
    generalParams = pd.read_csv("./data/general_parameters.csv",header=None)
    generalParamsNames = generalParams.iloc[:,0].values
    generalParamsValues = generalParams.iloc[:,1].values
    allParams['generalParamsNames'] = generalParamsNames
    allParams['generalParamsValues'] = generalParamsValues
    # unit Parameters
    unitParams = pd.read_csv("./data/unit_parameters.csv")
    unitParamsNames = unitParams.iloc[:,0].values
    unitParamsValues = unitParams.iloc[:,1:].values
    allParams['unitParamsNames'] = unitParamsNames
    allParams['unitParamsValues'] = unitParamsValues
    # internal patient transfer rate
    internalTransferRate = np.array(pd.read_csv("./data/patient_transfer_internal.csv",index_col=[0]))
    allParams['internalTransferRate'] = internalTransferRate
    # device transfer rate
    deviceTransferRate = np.array(pd.read_csv("./data/device_transfer_rate.csv",index_col=[0]))
    allParams['deviceTransferRate'] = deviceTransferRate
    # initial conditions
    initialConditions = pd.read_csv("./data/initial_conditions.csv")
    initialConditionsParams = initialConditions.iloc[:,0].values
    initialConditionsValues = initialConditions.iloc[:,1:].values
    allParams['initialConditionsParams'] = initialConditionsParams
    allParams['initialConditionsValues'] = initialConditionsValues
    # fitting data
    fitDatafiles = glob.glob("./data/INTUnit*")
    S_fitData = []
    X_fitData = []
    C_fitData = []
    I_fitData = []
    for file in fitDatafiles:
        fitData = pd.read_csv(file)
        S_fitData.append(fitData.loc[:,'Susceptible'].values)
        X_fitData.append(fitData.loc[:,'SusceptibleX'].values)
        C_fitData.append(fitData.loc[:,'Colonized'].values)
        I_fitData.append(fitData.loc[:,'Infected'].values)
    allParams['fitData'] = {'S_fitData': S_fitData, 'X_fitData': X_fitData, 'C_fitData': C_fitData, 'I_fitData': I_fitData}    
    return allParams


def setupModel(allParams):
    T = 100 * 24 #hourly
    S0, X0, UC0, DC0, I0, M0, Y0, L0, W0 = allParams['initialConditionsValues']
    P = S0 + X0 + UC0 + DC0 + I0
    N0_0 = P * allParams['unitParamsValues'][np.where(allParams['unitParamsNames']=='r_n')][0]
    N1_0 = np.zeros(len(P))
    D0_0 = P * allParams['unitParamsValues'][np.where(allParams['unitParamsNames']=='r_d')][0]
    D1_0 = np.zeros(len(P))
    y0 = np.hstack([S0, X0, UC0, DC0, I0, M0, N0_0, N1_0, D0_0, D1_0, Y0, L0, W0])
    tVector = np.linspace(0, T-1, T)
    return (y0, tVector)
    
    
    
    
    
    
    
