import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# use all cores
#import os
#os.system("taskset -p 0xff %d" % os.getpid())
pd.options.mode.chained_assignment = None # deactivating slicing warns

def load_seattle_speed_matrix():
    """ Loads the whole Seattle `speed_matrix_2015` into memory.
    Caution: ~ 200 mb of data
    
    :param: 
    :return df: speed matrix as a pandas.DataFrame, columns are sensors, rows are timestamps
    """ 
    speed_matrix = './data/Seattle_Loop_Dataset/speed_matrix_2015'
    print('Loading data...')
    df = pd.read_pickle(speed_matrix)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M')
    print('Load completed.')
    return df

def best_moving_average(df, col, average_window_in_hours=27, from_date=None, to_date=None, plot=False):
    """ Calculates the moving average in a window of `average_window_in_hours` hours and propagates
    into the future.
    
    Beware! This code uses data from the future to perform predictions.
    Meaning it is meant to be used to generate the "perfect" moving average baseline.
    
    :param df: dataset being used
    :param col: column for which the moving average will be applied
    :param average_window_in_hours: the window (in hours) used to generate predictions
    :param from_date: initial date to be shown in the plot
    :param to_date: end date to be shown in the plot
    :param plot: plot moving average and original df
    :return (MAE, RMSE): Both metrics are calculated for the column `col`
    """ 
    ndf = df[[col]]
    window_size = average_window_in_hours*12
    ndf['preds'] = ndf.rolling(window=window_size).mean().shift(1)
    MAE = ndf.apply((lambda x: np.abs(x[0] - x[1])), axis=1).dropna().mean()
    RMSE = np.sqrt(ndf.apply((lambda x: np.power(x[0] - x[1], 2)), axis=1).dropna().mean())
    if plot:
        if from_date is not None and to_date is not None:
            ndf.resample('1h').mean().loc[from_date:to_date].plot(figsize=(12, 7))
        else:
            ndf.resample('1h').mean()[:500].plot(figsize=(12, 7))
        plt.show()
    return (MAE, RMSE)


def calculate_metrics(df, average_window_in_hours, verbose=5, save=True):
    """ Calculates MAE and RMSE for all columns of `df`, taking a sliding window of `average_window_in_hours` hours. 
    :param df (panads.DataFrame): dataset being used
    :param average_window_in_hours (int): the window (in hours) used to generate predictions
    :param verbose (int): option to display the calculations on-the-fly.
        Values are going to be displayed after `verbose` iterations.
                    
    :return mae_and_rmse (dict): dictionary containing (MAE, RMSE) for each column of `df`
    """ 
    mae_and_rmse = {}
    for (it, col) in enumerate(df.columns):
        MAE, RMSE = best_moving_average(df, col, average_window_in_hours)
        mae_and_rmse[col] = (MAE, RMSE)
        if it%verbose == 0:
            print('Column: {}, MAE: {}, RMSE: {}'.format(col, MAE, RMSE))
    if save:
        pd.DataFrame(mae_rmse, index=['MAE', 'RMSE']).to_csv('./experiment_results/seattle_best_moving_average_mae_rmse.csv')
    return mae_and_rmse


def main():
    # TODO: setup argument parser
    SLIDING_WINDOW_IN_HOURS = 27
    SAVE_METRICS = True
    VERBOSE = 5
    df = load_seattle_speed_matrix()
    mae_and_rmse_dict = calculate_metrics(df, SLIDING_WINDOW_IN_HOURS, VERBOSE, SAVE_METRICS)

    
if __name__ == '__main__':
    main()
