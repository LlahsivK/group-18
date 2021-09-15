import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import datetime
from matplotlib import pyplot as plt
from scipy import stats
from itertools import product
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm_notebook
import warnings                                  # do not disturbe mode


def read_csv(file1):
    with open(file1, 'r') as f:
        reader = pd.read_csv(f)
    ceilometer = reader[['# Time', 'bl_height']]
    return ceilometer

def read_all(cl51):
    df = pd.DataFrame(columns = ['# Time', 'bl_height'])
    for i in range(len(cl51)):
        file1 = 'E:/' + cl51[i] + '.csv'
        ceilometer = read_csv(file1)
        df = pd.concat([df, ceilometer], ignore_index=True)
    return df

def data_process():
    cl51_Lidcombe = ['0212_Lidcombe', '0213_Lidcombe', '0214_Lidcombe', '0215_Lidcombe',
                     '0216_Lidcombe', '0217_Lidcombe', '0218_Lidcombe']
    Lidcombe = read_all(cl51_Lidcombe)
    cl51_df_Lidcombe = Lidcombe.replace(-999, np.nan)
    cl51_df_Lidcombe['datetime'] = pd.to_datetime(cl51_df_Lidcombe['# Time'], format='%d/%m/%Y %H:%M:%S')
    cl51_df_Lidcombe = cl51_df_Lidcombe.set_index('datetime')
    cl51_df_Lidcombe = cl51_df_Lidcombe.assign(revised_bl_height=cl51_df_Lidcombe.bl_height.interpolate(method='time'))
    return cl51_df_Lidcombe

def acfpacf(train):
    # acf pacf
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(train, lags=60, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(train, lags=60, ax=ax2)
    plt.savefig('./acfpacf.jpg')

# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# MAE RMSE IOA DTW-Distance MASE MBE
def satistical_comps_on_df(y_true, y_pred, y_train):
    # ioa = cohen_kappa_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ioa = index_agreement(y_pred, y_true)
    DTW = DTW_dist(y_true, y_pred)
    MASE = mase(y_train, y_true, y_pred)
    MBE= mbe(y_true,y_pred)
    return rmse, mae, ioa, DTW, MASE, MBE

def DTW_dist(y_true, y_pred):
    distance, path = fastdtw(y_true, y_pred, dist=euclidean)
    return distance, path

def mbe(y_true, y_pred):
    return np.mean(y_pred - np.mean(y_pred))

def mase(y_train, y_true, y_pred):
    n = y_train.shape[0]
    naive_error = np.sum(np.abs(np.diff(y_train)))/(n-1)
    model_error = np.mean(np.abs(y_true-y_pred))
    return model_error/naive_error


def index_agreement(s, o):
    """
    index of agreement

    Willmott (1981, 1982)
    input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    ia = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    return ia

def optimizeSARIMA(parameters_list, d, D, s, train):
    """Return dataframe with parameters and corresponding AIC

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order
        s - length of season
    """

    results = []
    best_aic = float("inf")

    for param in parameters_list:
        # we need try-except because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(train, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        print(aic)
        print(best_aic)
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table


def plotSARIMA(series, model, n_steps, real_data):
    """Plots model vs predicted values

        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
    """

    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['sarima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['sarima_model'][:24 + 1] = np.NaN

    real_df = pd.DataFrame(real_data)

    train_end = '2021-02-18 09:00:00'
    test_start = '2021-02-18 10:00:00'
    test_end = '2021-02-19 09:00:00'

    train_data = real_df[:train_end]
    test_data = real_df[test_start:test_end]

    #     # forecasting on n_steps forward
    #     forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    #     forecast = data.sarima_model.append(forecast)

    prediction = []

    for n in range(n_steps):
        model_n = sm.tsa.statespace.SARIMAX(train_data.revised_bl_height, order=(2, 1, 2),
                                            seasonal_order=(0, 1, 1, 24)).fit(disp=-1)
        # This is one time step forecast
        output = model_n.forecast()

        # Predicted value append
        yhat = output[0]
        prediction.append(yhat)

        # Basically append new observations as they become available.
        # Technically, this trains new model for every new history point.
        # So we are doing a 1 hour prediction - 24/25 times
        obs = test_data.iloc[n]
        train_data = train_data.append(obs)

    results_df = test_data.copy(deep=True)

    results_df['Predicted'] = prediction

    rmse, mae, ioa, DTW, MASE, MBE = satistical_comps_on_df(results_df['revised_bl_height'],
                                                            results_df['Predicted'],
                                                            real_df[:train_end])

    print("rmse", rmse)
    print("mae", mae)
    print("ioa", ioa)
    print("DTW", DTW)
    print("MASE", MASE)
    print("MBE", MBE)
    ia = index_agreement(results_df['revised_bl_height'], results_df['Predicted'])

    #     print(results_df)
    #     plt.figure(figsize=(15, 7))
    ax = results_df.plot(figsize=(15, 7))
    #     plt.plot(forecast, color='r', label="model")
    plt.title("IOA: {0:.5f}".format(ia))
    #     plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    #     plt.plot(real_data, label="actual")

    plt.legend()
    plt.grid(True)
    plt.savefig('./Lprediction.jpg')


def ARIMA_model(data):
    data_minute = data['revised_bl_height'].resample('1H').mean().round(3)
    # split train and test
    train = data_minute['2021-02-12':'2021-02-18 09:00:00']
    train = pd.DataFrame(train)
    train.plot(figsize=(12, 8))
    plt.title("cl51 Lidcombe train")
    plt.savefig('./bl_height.jpg')
    plt.show()

    # seasonal model test
    decomposition = sm.tsa.seasonal_decompose(train.revised_bl_height, model='additive', extrapolate_trend='freq')
    plt.rc('figure', figsize=(12, 8))
    fig = decomposition.plot()
    plt.savefig('./seasonaltest.jpg')

    # The seasonal difference
    #     blh_diff = train.revised_bl_height - train.revised_bl_height.shift(24)
    #     acfpacf(blh_diff[24:])

    #     #the first difference
    #     blh_diff = blh_diff - blh_diff.shift(1)
    #     acfpacf(blh_diff[24+1:])

    # setting initial values and some bounds for them
    ps = range(2, 5)
    d = 1
    qs = range(2, 5)
    Ps = range(0, 2)
    D = 1
    Qs = range(0, 2)
    s = 24  # season length is still 24

    # creating list with all the possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)

    '''
    warnings.filterwarnings("ignore")
    result_table = optimizeSARIMA(parameters_list, d, D, s, train.revised_bl_height)

    p, q, P, Q = result_table.parameters[0]
    print(p, q, P, Q)
    '''

    best_model = sm.tsa.statespace.SARIMAX(train.revised_bl_height, order=(2, 1, 2),
                                           seasonal_order=(0, 1, 1, 24)).fit(disp=-1)

    plotSARIMA(train, best_model, 24, data_minute)



if __name__ == '__main__':
    cl51_Lidcombe=data_process()
    ARIMA_model(cl51_Lidcombe)
