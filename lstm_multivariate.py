# multivariate input lstm poc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import math
import os
import time

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def read(filepath):
    data = pd.read_csv(filepath, 
                    usecols=['Date', 'Open', 'Close', 'Volume'])
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data
data = read('ENB.csv')
# print(data.head)

# time_shift = 2

# target_data = data['Open'].shift(time_shift)
# train_data = data.iloc[:-time_shift]
# print(target_data)
# print(train_data)


# test_head = data.index[int(0.8*len(data))]
# print(test_head)

def normalize(df):
    mean = df.mean()
    stdev = df.std()
    return (df-mean)/stdev

# dataframe, col 1, col 2, col1shiftback -> create Z-score(% change) col
# (x2-x1)/x1 should approx normal distribution
def calc_p_delta_Z(df, x1, x2, x1shift:int=0):
    temp_df = (df[x2] - df[x1].shift(x1shift)) / df[x1].shift(x1shift)
    df[x1+'_'+x2+'_Z'] = normalize(temp_df)

# dataframe, column -> create Z-score(ln(col)) col
# ln(x) should approx normal distribution
def calc_log_Z(df, x):
    temp_df = np.log(df[x])
    df['ln_'+x+'_Z'] = normalize(temp_df)

# attempt to normalize values for LSTM 
def process_df(df):
    calc_p_delta_Z(data, 'Open', 'Close')
    calc_p_delta_Z(data, 'Close', 'Open', x1shift=1)
    calc_log_Z(data, 'Volume')
    data.dropna(inplace=True)

process_df(data)

train_test_split = 0.9
split_index = int(len(data) * train_test_split)
df_train = data[:split_index]
df_test = data[split_index:]

# print(data)


class SequenceDataset(Dataset):
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




lookback = 7
target = 'Open_Close_Z'
delay_features = ['ln_Volume_Z']
features = ['Close_Open_Z']

train_dataset = SequenceDataset(
    df_train,
    target = target,
    features = features,
    delay_features = delay_features,
    lookback = lookback
)
test_dataset = SequenceDataset(
    df_test,
    target = target,
    features = features,
    delay_features = delay_features,
    lookback = lookback
)

batch_size = 3 # how come 3??? so low?
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# X, y = next(iter(train_loader))
# print(X)
# print("Features shape:", X.shape)
# print(y)
# print("Target shape:", y.shape)

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
        # print(h0, c0)

        # out, _ = self.lstm(x, (h0, c0)) # _ is updated tupple (h, c)
        # out = self.fc(out[:, -1, :]) # makes sense of the LSTM raw output, which is its state iirc
        # return out

        # alternative return out code
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out

def countSignMatch(x, y):
    assert x.size(dim=0) == y.size(dim=0)
    xy = x*y
    z = torch.zeros((x*y).shape)
    return torch.count_nonzero(torch.gt(xy,z))

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
        factor = m((numSignInc / numSignCor)-1.3)

        res = errorSq * factor / N 

        assert not math.isnan(res)
        return res


model = LSTM(num_features=1+len(delay_features+features), hidden_layer_size=128, num_layers=2)

learning_rate = 0.001
num_epochs = 200
loss_function = myLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model(data_loader, model, loss_function, optimizer):
    total_loss = 0.0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Train loss: {avg_loss}")

def test_model(data_loader, model, loss_function):
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / len(data_loader)
    print(f"Test loss: {avg_loss}")


for ix_epoch in range(30):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    test_model(test_loader, model, loss_function)
    print()


