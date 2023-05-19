import numpy as np
import torch

########## ANALYSE RAW DATA ##########

def calc_naive(df):
    n = df['Open_d_Close'].size
    print("+", sum(df['Open_d_Close'] > 0)/n*100)
    print("0", sum(df['Open_d_Close'] == 0)/n*100)
    print("-", sum(df['Open_d_Close'] < 0)/n*100)

    
    c = (df['Open_d_Close'] * df['Open_d_Close'].shift(1))[1:]
    print("Naive guess success rates:")
    print("correct", sum(c>0)/c.size * 100)
    print("incorrect", sum(c<=0)/c.size * 100)


    c = (c>0).astype(int) - (c<0).astype(int)
    print("Naive ROI:")
    r = df['Open_d_Close'][1:] * c + 1
    print(r.cumprod().iloc[-1])

    print()
    return

########## PROCESS RAW DATA ##########

def normalize(df):
    mean = df.mean()
    stdev = df.std()
    return (df-mean)/stdev

# dataframe, col 1, col 2, col1shiftback -> create Z-score(% change) col
# (x2-x1)/x1 should approx normal distribution
def calc_p_delta_Z(df, x1, x2, x1shift:int=0):
    df[x2+'_d_'+x1] = (df[x2] - df[x1].shift(x1shift)) / df[x1].shift(x1shift)
    df[x1+'_'+x2+'_Z'] = normalize(df[x2+'_d_'+x1])

# dataframe, column -> create Z-score(ln(col)) col
# ln(x) should approx normal distribution
def calc_log_Z(df, x):
    temp_df = np.log(df[x])
    df['ln_'+x+'_Z'] = normalize(temp_df)



########## RUNTIME DATA ##########

# do i need a with torch no grad wrapper and deep copy everything b4hand
def countSignMatch(x, y):
    assert x.size(dim=0) == y.size(dim=0)
    xy = x*y
    z = torch.zeros((x*y).shape)
    return torch.count_nonzero(torch.gt(xy,z))

# output tensor prediction, target, full processed data frame
def validate_output(yHat, y, df):
    assert yHat.size() == y.size()
    with torch.no_grad():

        sign_correct_proportion = countSignMatch(yHat, y).item() / y.size(dim=0)
        
        prod = yHat * y
        res = np.array((prod>0)).astype(int) - np.array((prod<0)).astype(int)
        res = np.reshape(res, res.size)
        deltas = df['Open_d_Close'].tail(len(res))
        rois = res * deltas + 1
        prod_roi = rois.cumprod().iloc[-1]

        ############## TODO: I WANT TO KNOW HOW MANY GUESSES ARE POSITIVE
        # i think theyre mostly positive
        # also check if im lagging behind. especially on spikes. 

    return sign_correct_proportion, prod_roi


########## ANALYSE RESULTING DATA ##########