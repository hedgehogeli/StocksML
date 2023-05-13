import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy as dc

import lstb_poc as lstb


device = lstb.device

selection = 0
CSVs = ['ENB', 'MFC', 'SU', 'XEG', 'XFN', 'XIU', 'GSPTSE']
dir = 'stock_data/'
write_dir = 'plot_outputs/'

lookback = 6



def graph_training_data_cmp():
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()
    plt.plot(y_train, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Day')
    plt.legend()
    plt.savefig(write_dir + csv + '_training_cmp.png')
    plt.clf()
    return

def graph_test_data_cmp():
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = test_predictions

    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = y_test.flatten()

    new_y_test = dc(dummies[:, 0])

    plt.plot(new_y_test, label='Actual')
    plt.plot(test_predictions, label='Predicted')
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
    
csv = CSVs[selection]
# for csv in CSVs:
print('*********** STARTING TRAINING ON STOCK', csv, '***********')
data = pd.read_csv(dir + csv + '.csv')
X_train, y_train, X_test, y_test, df_results = lstb.run_stock(data, lookback_days=lookback) 
# df_results: epoch avgTrainLoss testLoss testSignCorrect ROI
df_results.to_csv(write_dir + csv + '_results.csv')

model = lstb.model
graph_stuff()

print('*********** FINISHED WITH', csv, '***********')