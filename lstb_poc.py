import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from copy import deepcopy as dc
import time

import process_data


################################################################################
train_test_split = 0.9
################################################################################
batch_size = 16 
lookback = 7
target = 'Open_Close_Z'
delay_features = ['ln_Volume_Z']
features = ['Close_Open_Z']
################################################################################
hiddenSize = 128 
lstmLayers = 1
################################################################################
learning_rate = 0.0002
num_epochs = 200
################################################################################

device = 'cpu'

class LSTM(nn.Module):
    def __init__(self, num_features, hidden_layer_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(num_features, hidden_layer_size, num_layers, batch_first=True)
        
        # fc_1 optional? remove to prevent overfit to train data?
        # self.fc_1 =  nn.Linear(hidden_layer_size, hidden_layer_size) # fully connected
        self.fc = nn.Linear(hidden_layer_size, 1) # fully connected last layer

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(device)

        # out, _ = self.lstm(x, (h0, c0)) # _ is updated tupple (h, c)
        # out = self.fc(out[:, -1, :]) # makes sense of the LSTM raw output, which is its state iirc
        # return out

        # alternative return out code
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out  

# CUSTOM LOSS FUNCTION
# L = SUM { ln(pred-target)**2 + 1 } * tanh(#inc/#cor)
class myLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yHat, y):
        errorSq = torch.sum(torch.log(torch.square(yHat-y)+1))
        ### WANT: yHat-y weighted differently according to sign correctness
        n = y.size(dim=0)
        N = torch.tensor(n)
        
        numSignCor = process_data.countSignMatch(yHat, y)
        numSignInc = N - numSignCor
        m = nn.Sigmoid()
        factor = m((numSignInc / numSignCor)-1.3)

        res = errorSq * factor / N 

        assert not math.isnan(res)
        return res
    
model = LSTM(num_features = 1 + len(delay_features+features), 
             hidden_layer_size = hiddenSize, 
             num_layers = lstmLayers)
model.to(device)

loss_function = myLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.70)

class StockDataset(Dataset):
    def __init__(self, df, target, features, delay_features, lookback=7):
        self.target = target
        self.features = features # features known on present day
        self.delay_features = delay_features # features not yet known
        self.lookback = lookback
        self.y = torch.tensor(df[target].values).float()
        self.X = torch.tensor(df[delay_features + features].values).float()
        self.length = self.X.shape[0] - self.lookback - 1 # 1 for delayed
    
    def __len__(self):
        return self.length

    def __getitem__(self, i): 
        i_end = i+self.lookback
        features = self.X[i+1:(i_end+1), len(self.delay_features):]
        delay_features = self.X[i:i_end, :len(self.delay_features)]
        prev_targets = torch.reshape(self.y[i:i_end], (self.lookback, 1))
        x = torch.cat((prev_targets, delay_features, features), dim=1)
        return x, self.y[i_end]
    
def run_stock(data):

    prepped_data = process_df(data)
    process_data.calc_naive(prepped_data)

    split_index = int( len(prepped_data) * train_test_split )
    df_train = prepped_data[:split_index]
    df_test = prepped_data[split_index:]
    train_dataset = StockDataset(
        df_train,
        target = target, features = features, delay_features = delay_features,
        lookback = lookback )
    test_dataset = StockDataset(
        df_test,
        target = target, features = features, delay_features = delay_features,
        lookback = lookback )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # why shuffle every epoch?
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_train_loader = DataLoader(train_dataset, batch_size=len(df_train), shuffle=False)
    full_test_loader = DataLoader(test_dataset, batch_size=len(df_test), shuffle=False)

    # for _, batch in enumerate(train_loader):
        # x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # print(x_batch.shape, y_batch.shape) # torch.Size([16, 17, 1]) torch.Size([16, 1])
        # break # this only runs once???

  
    # TRAIN MODEL
    start_time = time.time()
    epoch_stats_df_row_list = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(train_loader)
        test_loss, output, pred_target = validate_one_epoch(test_loader, full_test_loader)
        scheduler.step()
        
        # collect epoch info
        sign_correct, roi = process_data.validate_output(output, pred_target, prepped_data) # move to process_data
        epoch_stats_df_row_list.append({'epoch':epoch, 'avgTrainLoss':train_loss, 'testLoss':test_loss, 
                        'testSignCorrect':sign_correct, 'ROI':roi})
        
        if not epoch % 30:
            log_train_speed(epoch, start_time)

    return prepped_data, full_train_loader, full_test_loader, pd.DataFrame(epoch_stats_df_row_list)


# attempt to normalize values for LSTM 
def process_df(df):
    process_data.calc_p_delta_Z(df, 'Open', 'Close')
    process_data.calc_p_delta_Z(df, 'Close', 'Open', x1shift=1)
    process_data.calc_log_Z(df, 'Volume')
    df.dropna(inplace=True)
    return df

def log_train_speed(epoch, start_time):
    seconds_elapsed = (time.time() - start_time)
    seconds_per_epoch = seconds_elapsed / (epoch+1)
    est_time_remaining = (num_epochs-(epoch+1))*seconds_per_epoch

    print('epoch', epoch, '/', num_epochs)
    print('time elapsed: {0:.2f}s'.format(seconds_elapsed))
    print('ETA: {0:.2f}s'.format(est_time_remaining))
    print()



# return average loss over epoch
def train_one_epoch(train_loader):
    model.train(True)

    loss_accumulator = 0.0
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss_accumulator += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_accumulator / len(train_loader)

# return (avg loss, output tensor, target tensor) for entire test period
def validate_one_epoch(test_loader, full_test_loader):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    ret_full_output = None
    ret_target = None
    for batch_index, batch in enumerate(full_test_loader):
        x_batch, ret_target = batch[0].to(device), batch[1].to(device)
        ret_full_output = model(x_batch)
        break # should only run once anyway

    avg_loss = running_loss / len(test_loader)
    return avg_loss, ret_full_output, ret_target


