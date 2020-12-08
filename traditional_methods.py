import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# use all cores
#import os
#os.system("taskset -p 0xff %d" % os.getpid())
pd.options.mode.chained_assignment = None # deactivating slicing warns

def load_seattle_speed_matrix():
    """ Loads the whole Seattle `speed_matrix_2015` into memory.
    Caution ~ 200 mb of data
    
    :param: 
    :return df (pandas.DataFrame): speed matrix as DataFrame. Columns are sensors, rows are timestamps
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
    
    :param df (pandas.DataFrame): dataset being used
    :param col (str): column for which the moving average will be applied
    :param average_window_in_hours (int): the window (in hours) used to generate predictions
    :param from_date (str): initial date to be shown in the plot, format: "YYYY-MM-DD"
    :param to_date (str): end date to be shown in the plot
    :param plot (bool): plot moving average and original df
    :return MAE, RMSE (tuple): Both metrics are calculated for the column `col`
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
    :param save (bool): 
    :return mae_and_rmse (dict): dictionary containing (MAE, RMSE) for each column of `df`
    """ 
    mae_and_rmse = {}
    for (it, col) in enumerate(df.columns):
        MAE, RMSE = best_moving_average(df, col, average_window_in_hours)
        mae_and_rmse[col] = (MAE, RMSE)
        if it%verbose == 0:
            print('Column: {}, MAE: {}, RMSE: {}'.format(col, MAE, RMSE))
    if save:
        # TODO: add param to attribute filename and filedir
        pd.DataFrame(mae_rmse, index=['MAE', 'RMSE']).to_csv('./experiment_results/seattle_best_moving_average_mae_rmse.csv')
    return mae_and_rmse



def real_moving_average(df, col, sliding_window_in_hours, forecast_window_in_minutes):
    """ Calculating the moving average using a sliding window of `sliding_window_in_hours` 
    on a forecast window of `forecast_window_in_minutes` over the dataset. 
    Returns a dataframe with the forecast for the given dataframe.
    """
    sliding_window = 12*sliding_window_in_hours
    forecast_window = ((forecast_window_in_minutes+5)//5)
    
    X = df[col].values
    Y = X[:sliding_window]
    
    for i in range(forecast_window):
        ypred = np.mean(Y[i: i+sliding_window])
        Y = np.append(Y, ypred)
    forecast_df = pd.DataFrame(
        data=Y[len(Y)-forecast_window:], 
        index=df.index[sliding_window:sliding_window+forecast_window]
    )
    return forecast_df

# still need to compute MAE and RMSE for all data
def moving_average_forecast(df, col, sliding_window_in_hours, forecast_window_in_minutes):
    """ Applies moving average forecast across all the dataset. Stride can be applied to make forecasting faster, 
    ie, stride makes the sliding window jump a window of `stride_in_minutes`.
    
    Returns a pandas.DataFrame containing a side-by-side comparison of the real dataframe and its predictions, 
    for all predicted values.
    """
    sliding_window = 12*sliding_window_in_hours
    forecast_window = ((forecast_window_in_minutes+5)//5)
    stride_in_minutes = 60
    stride = (stride_in_minutes//5)
    
    all_predictions = []
    
    if stride_in_minutes == 0:
        max_it = len(df)
    else:
        max_it = len(df)//stride
        
    for i in range(max_it):
        try:
            smaller_df = df.iloc[i*stride: (sliding_window+forecast_window) + (i+1)*stride]
            preds = real_moving_average(smaller_df, col, sliding_window_in_hours, forecast_window_in_minutes)
            fdf = pd.concat([smaller_df[[col]].loc[preds.index[0]:preds.index[-1]],preds], axis=1)
            fdf = fdf.rename(columns={0:col+'_pred'})
            all_predictions.append(fdf)
        except:
            pass
    return pd.concat(all_predictions, axis=0)


def metrics(preds_df):
    """ Given a `preds_df` containing two columns, the first with real values and the second being preds,
    returns MAE and RMSE

    """
    preds = preds_df
    MAE = np.mean(np.abs(preds[preds.columns[0]] - preds[preds.columns[1]] ))
    RMSE = np.sqrt(np.mean(np.power(preds[preds.columns[0]] - preds[preds.columns[1]], 2)))
    return (MAE, RMSE)


def main():
    # this options should go into an argument parser
    SLIDING_WINDOW_IN_HOURS = 2
    FORECAST_WINDOW_IN_MINUTES = 15
    STRIDE_IN_MINUTES = 60
    
    metrics_dict = {}
    for col in df.columns:
        print(col)
        preds = moving_average_forecast(df, col, SLIDING_WINDOW_IN_HOURS, FORECAST_WINDOW_IN_MINUTES)
        mae_rmse = metrics(preds)
        metrics_dict[col] = mae_rmse
        
    pd.DataFrame(metrics_dict, index=['MAE', 'RMSE']).to_csv('./experiment_results/15_min_mae_rmse_seattle.csv')
    
if __name__ == '__main__':
    main()