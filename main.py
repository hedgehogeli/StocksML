import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from copy import deepcopy as dc

import lstb_poc as lstb


device = lstb.device

selection = 0
CSVs = ['ENB', 'MFC', 'SU', 'XEG', 'XFN', 'XIU', 'GSPTSE']
dir = 'stock_data/'
write_dir = 'plot_outputs/'

    


def graph_training_data_cmp():
    with torch.no_grad():
        X, y = next(iter(full_train_loader))
        ret_full_output = model(X)
    plt.plot(y, label='Actual')
    plt.plot(ret_full_output, label='Predicted')
    plt.xlabel('Day')
    plt.legend()
    plt.savefig(write_dir + csv + '_training_cmp.png')
    plt.clf()
    return

def graph_test_data_cmp():
    with torch.no_grad():
        X, y = next(iter(full_test_loader))
        ret_full_output = model(X)
    plt.plot(y, label='Actual')
    plt.plot(ret_full_output, label='Predicted')
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel('Day')
    plt.legend()
    plt.savefig(write_dir + csv + '_test_cmp.png')
    plt.clf()
    return

def graph_model_progression():
    plt.plot(df_results['avgTrainLoss'], label='training losses', color='b')
    plt.plot(df_results['testLoss'], label='test losses', color='y')
    plt.plot(df_results['testSignCorrect'], label='test sign correctness', color='g')
    plt.plot(df_results['ROI'], label='rois', color='m')
    plt.axhline(y=1.0, color='r', linestyle='dotted')
    plt.axhline(y=0.5, color='r', linestyle='dotted')
    plt.axhline(y=0.55, color='r', linestyle='dotted')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(write_dir + csv + '_progression.png')
    plt.clf()
    return

def graph_stuff():
    graph_training_data_cmp()
    graph_test_data_cmp()
    graph_model_progression()
    return

def analyse_results():
    # % of guesses were positive / negative
    # am i lagging behind spikes? how to check numerically?

    return

def read(filepath):
    data = pd.read_csv(filepath, 
                    usecols=['Date', 'Open', 'Close', 'Volume'])
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data
    
csv = CSVs[selection]
# for csv in CSVs:
print('*********** STARTING TRAINING ON STOCK', csv, '***********')

data = read(dir + csv + '.csv')
prepped_data, full_train_loader, full_test_loader, df_results = lstb.run_stock(data) 
# df_results: epoch avgTrainLoss testLoss testSignCorrect ROI
df_results.to_csv(write_dir + csv + '_results.csv')

split_index = int( len(prepped_data) * lstb.train_test_split )
df_train = prepped_data[:split_index]
df_test = prepped_data[split_index:]

model = lstb.model
graph_stuff()

print('*********** FINISHED WITH', csv, '***********')