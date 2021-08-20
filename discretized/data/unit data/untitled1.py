import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

cont = glob.glob('INTUnit*')
for file in cont[2:]:
    data = pd.read_csv(file)
    data['transfer_in'] = 0
    data['transfer_in'] = data[['transfer','transfer_in']].max(axis=1)
    data['transfer_out'] = 0
    data['transfer_out'] = data[['transfer','transfer_out']].min(axis=1)
    data['transfer_out'] = data['transfer_out'].abs()
    data.drop(columns=['transfer'], inplace=True)
    data.to_csv(file)