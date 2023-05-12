# Basic ENB Test Run

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

device = 'cpu'

lookback = 5 # num of days LSTM looks backwards
train_test = 0.9 # proportion of train/test. currently 5 years, 4.5 train 0.5 test

CSVs = ['ENB.csv', 'SU.csv', 'MFC.csv', 'XEG.csv', 'XFN.csv', 'XIU.csv', 'GSPTSE.csv']

dir = 'stock_data/'
data = pd.read_csv(dir+'ENB.csv')
# data = pd.read_csv('SU.csv')
# data = pd.read_csv('MFC.csv')

data.drop(columns=['High', 'Low', 'Adj Close'], axis=1, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])



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


def prep_lstm_dataframe(df, days_back):
    
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

prepped_data = prep_lstm_dataframe(data, lookback)
# print(prepped_data)
# prepped_data.to_csv('out.csv')






data_np = prepped_data.to_numpy()

y = data_np[:, 0] # 1st col 
X = data_np[:, 1:] # all cols excluding 1st (open_close_Z is our excluded b/c predicting it)
X = dc(np.flip(X, axis=1)) # reverses order of column, so most recent data is last, -5 -> -1 days
# print(X.shape, Y.shape)

split_index = int(len(X) * train_test)
# print(split_index)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# pytorch conversion
# lstm requires extra dimension
X_train = X_train.reshape((-1, lookback*3+2, 1))
X_test = X_test.reshape((-1, lookback*3+2, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# pytorch tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# pytorch dataset:
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
# print(train_dataset)


batch_size = 15 # why 16? what does this mean?
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # why shuffle every epoch?
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    # print(x_batch.shape, y_batch.shape)
    break
# output right now: torch.Size([16, 17, 1]) torch.Size([16, 1])



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


# do i need a with torch no grad wrapper and deep copy everything b4hand
def countSignMatch(x, y):
    assert x.size(dim=0) == y.size(dim=0)
    xy = x*y
    z = torch.zeros((x*y).shape)
    return torch.count_nonzero(torch.gt(xy,z))


# CUSTOM LOSS FUNCTION
# trying L = SUM{(pred-target)**2} * tanh(#inc/#cor)
class myLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yHat, y):
        errorSq = torch.sum(torch.log(torch.square(yHat-y)+1))
        ### use log????
        ### ln(SQUARE + 1)
        n = y.size(dim=0)
        N = torch.tensor(n)
        
        numSignCor = countSignMatch(yHat, y)
        numSignInc = N - numSignCor
        m = nn.Sigmoid()
        factor = m((numSignInc / numSignCor)-1)
        ### weighted buckets during count, diff weigts for 2 sides
        ### use a function instead of weighted buckets


        res = errorSq * factor / N 

        assert not math.isnan(res)
        return res
    




avgTrainLosses = []
testLosses = []
testSignCorrectness = []
accSignCorrect = np.empty(0, dtype=np.int8)
ROIs = []

def train_one_epoch():
    model.train(True)
    # print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    accLoss = 0.0
    batchcount = 0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchcount += 1

        if not batch_index % 10:
            avg_loss_across_batches = running_loss / 10
            # if running_loss > 0.0:
            #     print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
            #                                             avg_loss_across_batches))
            accLoss += running_loss
            running_loss = 0.0
    
    avgTrainLosses.append(accLoss/batch_index)
    # print()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    global accSignCorrect
    signCorrect = 0
    itemCount = 0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        # print(batch_index)
        # print(y_batch)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            # accumulate +/- correctness 
            signCorrect += countSignMatch(output, y_batch).item()
            itemCount += output.size(dim=0)

            prod = output * y_batch
            res = np.array((prod>0)).astype(int) - np.array((prod<0)).astype(int)
            res = np.reshape(res, res.size)
            accSignCorrect = np.append(accSignCorrect, res)
    
    deltas = data['open_d_close'].tail(len(accSignCorrect))
    r = accSignCorrect * deltas + 1
    roi = r.cumprod().iloc[-1]
    ROIs.append(roi)

    ############## TODO: I WANT TO KNOW HOW MANY GUESSES ARE POSITIVE
    # i think theyre mostly positive
    # also check if im lagging behind. especially on spikes. 

    accSignCorrect = np.empty(0, dtype=np.int8)
    

    avg_loss_across_batches = running_loss / len(test_loader)
    
    # print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    # print('***************************************************\n')


    testLosses.append(avg_loss_across_batches)
    print("epoch: ", epoch, signCorrect, "/", itemCount)
    testSignCorrectness.append(signCorrect/itemCount)


################################################################################
################################################################################
# TODO: almost def need more than 4 hidden layers i think
hiddenSize = 64
lstmLayers = 2
################################################################################
model = LSTM(1, hiddenSize, lstmLayers) 
model.to(device)
################################################################################
################################################################################
# TODO: LearningRate_SCHEDULER
learning_rate = 0.001
num_epochs = 200
loss_function = myLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# LEARNING RATE SCHEDULER
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.40)



for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()
    scheduler.step()


atl = np.around(np.array(avgTrainLosses),5)
tl = np.around(np.array(testLosses),5)
tsc = np.around(np.array(testSignCorrectness),5)
rois = np.around(np.array(ROIs),5)
print(atl)
print(tl)
print(tsc)
print(rois)

plt.plot(atl, label='training losses', color='b')
plt.plot(tl, label='test losses', color='y')
plt.plot(tsc, label='test sign correctness', color='g')
plt.plot(rois, label='rois', color='m')
plt.axhline(y=1.0, color='r', linestyle='dotted')
plt.axhline(y=0.5, color='r', linestyle='dotted')
plt.axhline(y=0.55, color='r', linestyle='dotted')
plt.xlabel('epoch')
plt.show()











### GRAPHING ###


def graphStuff(): 
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

    plt.plot(y_train, label='Actual Close')
    plt.plot(predicted, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:, 0] = train_predictions

    train_predictions = dc(dummies[:, 0])
    train_predictions

    dummies = np.zeros((X_train.shape[0], lookback+1))
    dummies[:, 0] = y_train.flatten()

    new_y_train = dc(dummies[:, 0])
    new_y_train

    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = test_predictions

    test_predictions = dc(dummies[:, 0])
    test_predictions

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = y_test.flatten()

    new_y_test = dc(dummies[:, 0])
    new_y_test

    residuals = new_y_test - test_predictions

    plt.plot(new_y_test, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.plot(residuals, label='resids')

    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

graphStuff()