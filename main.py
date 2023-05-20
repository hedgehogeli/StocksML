import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

import time
from copy import deepcopy as dc

import lstb_poc as lstb
import process_data


device = lstb.device

selection = 1
CSVs = ['ENB', 'MFC', 'SU', 'XEG', 'XFN', 'XIU', 'GSPTSE']
dir = 'stock_data/'
write_dir = 'plot_outputs/'

num_backtest = 10

def graph_stuff():
    process_data.graph_training_data_cmp(model, full_train_loader, write_dir, csv, i)
    process_data.graph_test_data_cmp(model, full_test_loader, write_dir, csv, i)
    process_data.graph_model_progression(df_results, write_dir, csv, naive_success, naive_roi, i)
    return

def read(filepath):
    data = pd.read_csv(filepath, 
                    usecols=['Date', 'Open', 'Close', 'Volume'])
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def log_speed():
    seconds_elapsed = (time.time() - start_time)
    seconds_per_run = seconds_elapsed / runs
    est_time_remaining = (total_runs-runs)*seconds_per_run

    print('bt', i, '/', num_backtest)
    print('time elapsed: {0:.2f}s'.format(seconds_elapsed))
    print('ETA: {0:.2f}m'.format(est_time_remaining/60))
    print()

start_time = time.time()
total_runs = num_backtest * len(CSVs)

runs = 0
# csv = CSVs[selection]
for csv in CSVs:
    print('*********** STARTING TRAINING ON STOCK', csv, '***********')

    data = read(dir + csv + '.csv')
    
    for i in range(num_backtest):
        runs += 1

        print(f'***  BACKTEST {i+1} ***')
        data_subset = data.iloc[i : -(num_backtest-i)]

        prepped_data = lstb.process_df(data_subset)
        
        full_train_loader, full_test_loader, df_results = lstb.run_stock(prepped_data) 

        # df_results: epoch avgTrainLoss testLoss testSignCorrect ROI
        df_results.to_csv(write_dir + 'csvs/' + csv + f'_results_bt{i+1}.csv')

        split_index = int( len(prepped_data) * lstb.train_test_split )
        df_train = prepped_data[:split_index]
        df_test = prepped_data[split_index:]

        naive_success, naive_roi = process_data.calc_naive(df_test)

        model = lstb.model
        graph_stuff()

        log_speed()

    print('*********** FINISHED WITH', csv, '***********')