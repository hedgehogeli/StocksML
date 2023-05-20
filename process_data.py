import numpy as np
import torch
import matplotlib.pyplot as plt

########## ANALYSE RAW DATA ##########

def calc_naive(df):
    n = df['Open_d_Close'].size
    pos = sum(df['Open_d_Close'] > 0)/n*100
    zero = sum(df['Open_d_Close'] == 0)/n*100
    neg = sum(df['Open_d_Close'] < 0)/n*100
    print(f'open to close breakdown: + {pos} / 0 {zero} / - {neg}')
    # print("+", sum(df['Open_d_Close'] > 0)/n*100)
    # print("0", sum(df['Open_d_Close'] == 0)/n*100)
    # print("-", sum(df['Open_d_Close'] < 0)/n*100)

    
    c = (df['Open_d_Close'] * df['Open_d_Close'].shift(1))[1:]
    naive_success_rate = sum(c>0)/c.size
    print("Naive guess success rate:", naive_success_rate)
    # print("incorrect", sum(c<=0)/c.size * 100)


    c = (c>0).astype(int) - (c<0).astype(int)
    r = df['Open_d_Close'][1:] * c + 1
    naive_roi = r.cumprod().iloc[-1]
    print(f"Naive ROI: {naive_roi}")

    print()
    return naive_success_rate, naive_roi

########## PROCESS RAW DATA ##########

def normalize(df):
    mean = df.mean()
    stdev = df.std()
    return (df-mean)/stdev

# dataframe, col 1, col 2, col1shiftback -> create Z-score(% change) col
# (x2-x1)/x1 should approx normal distribution
def calc_p_delta_Z(df, x1, x2, x1shift:int=0):
    df = df.assign(delta = (df[x2] - df[x1].shift(x1shift)) / df[x1].shift(x1shift))
    df.rename(columns={'delta' : x2+'_d_'+x1}, inplace=True)
    # df.loc[x2+'_d_'+x1] = (df[x2] - df[x1].shift(x1shift)) / df[x1].shift(x1shift)
    df = df.assign(delta_Z = normalize(df[x2+'_d_'+x1])) 
    df.rename(columns={'delta_Z' : x2+'_'+x1+'_Z'}, inplace=True)
    # df.loc[x1+'_'+x2+'_Z'] = normalize(df[x2+'_d_'+x1])
    return df
    

# dataframe, column -> create Z-score(ln(col)) col
# ln(x) should approx normal distribution
def calc_log_Z(df, x):
    df = df.assign(log_Z = normalize(np.log(df[x])))
    df.rename(columns={'log_Z' : 'ln_'+x+'_Z'}, inplace=True)
    return df


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

def graph_training_data_cmp(model, input, write_dir, csv, i):
    with torch.no_grad():
        X, y = next(iter(input))
        ret_full_output = model(X)
    plt.plot(y, label='Actual')
    plt.plot(ret_full_output, label='Predicted')
    plt.xlabel('Day')
    plt.legend()
    plt.savefig(write_dir + csv + f'_training_cmp_bt{i+1}.png')
    plt.clf()
    return

def graph_test_data_cmp(model, input, write_dir, csv, i):
    with torch.no_grad():
        X, y = next(iter(input))
        ret_full_output = model(X)
    plt.plot(y, label='Actual')
    plt.plot(ret_full_output, label='Predicted')
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel('Day')
    plt.legend()
    plt.savefig(write_dir + csv + f'_test_cmp_bt{i+1}.png')
    plt.clf()
    return

def graph_model_progression(df_results, write_dir, csv, naive_success, naive_roi, i):
    plt.plot(df_results['avgTrainLoss'], label='training losses', color='b')
    plt.plot(df_results['testLoss'], label='test losses', color='y')
    plt.plot(df_results['testSignCorrect'], label='test sign correctness', color='g')
    plt.plot(df_results['ROI'], label='rois', color='m')
    plt.axhline(y=naive_roi, color='r', linestyle='dotted')
    plt.axhline(y=naive_success, color='b', linestyle='dotted')
    plt.axhline(y=0.55, color='r', linestyle='dotted')
    plt.axhline(y=0.5, color='r', linestyle='dotted')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(write_dir + csv + f'_progression_bt{i+1}.png')
    plt.clf()
    return

def analyse_results():

    return