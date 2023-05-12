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



device = 'cpu'

# TODO: my input_size = # of features should probably be 3 ???
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1) # last layer is 

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0)) # _ is updated tupple (h, c)
        out = self.fc(out[:, -1, :]) # makes sense of the LSTM raw output, which is its state iirc
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
        
        numSignCor = countSignMatch(yHat, y)
        numSignInc = N - numSignCor
        m = nn.Sigmoid()
        factor = m((numSignInc / numSignCor)-1)

        res = errorSq * factor / N 

        assert not math.isnan(res)
        return res

# TODO: 
################################################################################
################################################################################
hiddenSize = 64
lstmLayers = 2
model = LSTM(1, hiddenSize, lstmLayers) 
model.to(device)
################################################################################
learning_rate = 0.001
num_epochs = 200
loss_function = myLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.60)
################################################################################
################################################################################


def run_stock(data, lookback_days = 5, train_test_split = 0.9):
    data.drop(columns=['High', 'Low', 'Adj Close'], axis=1, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])

    prepped_data = normalize_dataframe(data, lookback_days)
    X_train, y_train, X_test, y_test = for_pytorch_dataframe(prepped_data.to_numpy(), lookback_days, train_test_split)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    batch_size = 15 # why 16? what does this mean?
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # why shuffle every epoch?
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    full_test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # print(x_batch.shape, y_batch.shape) # torch.Size([16, 17, 1]) torch.Size([16, 1])
        break # this only runs once???

  
    start_time = time.time()

    df_row_list = []
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(train_loader)
        test_loss, output, target = validate_one_epoch(test_loader, full_test_loader)
        scheduler.step()
        
        sign_correct, roi = validate_output(output, target, data)
        df_row_list.append({'epoch':epoch, 'avgTrainLoss':train_loss, 'testLoss':test_loss, 
                        'testSignCorrect':sign_correct, 'ROI':roi})
        
        if not epoch % 20:
            seconds_elapsed = (time.time() - start_time)
            seconds_per_epoch = seconds_elapsed / (epoch+1)
            est_time_remaining = (num_epochs-(epoch+1))*seconds_per_epoch

            print('epoch', epoch, '/', num_epochs)
            print('time elapsed: {0:.2f}s'.format(seconds_elapsed))
            print('ETA: {0:.2f}s'.format(est_time_remaining))
            print()


    return X_train, y_train, X_test, y_test, pd.DataFrame(df_row_list)


def calc_naive(df):
    n = df['open_d_close'].size
    print("+", sum(df['open_d_close'] > 0)/n*100)
    print("0", sum(df['open_d_close'] == 0)/n*100)
    print("-", sum(df['open_d_close'] < 0)/n*100)

    
    c = (df['open_d_close'] * df['open_d_close'].shift(1))[1:]
    print("Naive guess success rates:")
    print("correct", sum(c>0)/c.size * 100)
    print("incorrect", sum(c<=0)/c.size * 100)


    c = (c>0).astype(int) - (c<0).astype(int)
    print("Naive ROI:")
    r = df['open_d_close'][1:] * c + 1
    print(r.cumprod().iloc[-1])

    print()
    return

def normalize_dataframe(df, days_back):
    # calc yest_close->today_open and today_open->today_close % deltas
    df['open_d_close'] = (df['Close'] - df['Open']) / df['Open']  # (close-open)/open
    df['close_d_open'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) 

    # "normalize":
    # %delta -> Z-score 
    open_d_close_mean = df['open_d_close'].mean() 
    open_d_close_stdev = df['open_d_close'].std()
    df['open_close_Z'] = (df['open_d_close']-open_d_close_mean)/open_d_close_stdev

    close_d_open_mean = df['close_d_open'].mean() 
    close_d_open_stdev = df['close_d_open'].std()
    df['close_open_Z'] = (df['close_d_open']-close_d_open_mean)/close_d_open_stdev

    # volume -> log normal
    df['ln_vol'] = np.log(df['Volume'])
    ln_vol_mean = df['ln_vol'].mean()
    ln_vol_stdev = df['ln_vol'].std()
    df['ln_vol_Z'] = (df['ln_vol']-ln_vol_mean)/ln_vol_stdev
    
    # copy in prev lookback days
    for i in range(1, days_back+1):
        df[f'ln_vol_Z_-{i}'] = df['ln_vol_Z'].shift(i)
        df[f'close_open_Z_-{i}'] = df['close_open_Z'].shift(i)
        df[f'open_close_Z_-{i}'] = df['open_close_Z'].shift(i)

    calc_naive(df)

    df = dc(df)
    df.drop(columns=['Date', 'Volume', 'ln_vol', 'Open', 'Close', 'close_d_open', 'open_d_close'], 
            axis=1, inplace=True)
    df.dropna(inplace=True)
    
    return df

def for_pytorch_dataframe(df, lookback_days, train_test_split):
    y = df[:, 0] # 1st col 
    X = df[:, 1:] # all cols excluding 1st (open_close_Z is our excluded b/c predicting it)
    X = dc(np.flip(X, axis=1)) # reverses order of column, so most recent data is last, -5 -> -1 days

    split_index = int(len(X) * train_test_split)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # pytorch conversion
    # lstm requires extra dimension
    X_train = X_train.reshape((-1, lookback_days*3+2, 1))
    X_test = X_test.reshape((-1, lookback_days*3+2, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # pytorch tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    return X_train, y_train, X_test, y_test
        
def validate_output(yHat, y, df):
    # print(yHat.size)
    # print(y.size)
    assert yHat.size() == y.size()
    with torch.no_grad():

        sign_correct_proportion = countSignMatch(yHat, y).item() / y.size(dim=0)
        
        prod = yHat * y
        res = np.array((prod>0)).astype(int) - np.array((prod<0)).astype(int)
        res = np.reshape(res, res.size)
        deltas = df['open_d_close'].tail(len(res))
        rois = res * deltas + 1
        prod_roi = rois.cumprod().iloc[-1]

        ############## TODO: I WANT TO KNOW HOW MANY GUESSES ARE POSITIVE
        # i think theyre mostly positive
        # also check if im lagging behind. especially on spikes. 

    return sign_correct_proportion, prod_roi

# do i need a with torch no grad wrapper and deep copy everything b4hand
def countSignMatch(x, y):
    assert x.size(dim=0) == y.size(dim=0)
    xy = x*y
    z = torch.zeros((x*y).shape)
    return torch.count_nonzero(torch.gt(xy,z))



class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    


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

        # if not batch_index % 10:
            # avg_loss_across_batches = running_loss / 10
        #     print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
        #                                             avg_loss_across_batches))
            # running_loss = 0.0
    
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

    # print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    # print('***************************************************\n')












# runs the entire thing
# with torch.no_grad():
#         predicted = model(X_train.to(device)).to('cpu').numpy()




#     train_predictions = predicted.flatten()

#     dummies = np.zeros((X_train.shape[0], lookback+1))
#     dummies[:, 0] = train_predictions

#     train_predictions = dc(dummies[:, 0])
#     train_predictions

#     dummies = np.zeros((X_train.shape[0], lookback+1))
#     dummies[:, 0] = y_train.flatten()

#     new_y_train = dc(dummies[:, 0])
#     new_y_train

#     plt.plot(new_y_train, label='Actual Close')
#     plt.plot(train_predictions, label='Predicted Close')
#     plt.xlabel('Day')
#     plt.ylabel('Close')
#     plt.legend()
#     plt.show()
