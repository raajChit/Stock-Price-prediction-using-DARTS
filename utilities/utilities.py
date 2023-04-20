import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from functools import reduce
from operator import concat
import sys
import os

# getting the name of the directory
# where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path
sys.path.append(parent)


def get_preprocessed_datasets():
    train_set = pd.read_csv('./data/train_set.csv')
    val_set = pd.read_csv('./data/val_set.csv')
    test_set = pd.read_csv('./data/test_set.csv')
    return(train_set, val_set, test_set)


def get_target_covariate_timeseries(data_frame, target_columns):
    if not isinstance(data_frame, pd.DataFrame):
        raise Exception('ERROR: No dataframe passed')

    if target_columns == None:
        raise Exception('ERROR: No column names passed as target columns')

    targate_value_cols = []
    covariate_value_cols = []

    all_columns = data_frame.columns.values
    for col in all_columns:

        normalName = col
        if col not in target_columns and col != "Date":
            covariate_value_cols.append(normalName)
        elif col != "Date":
            targate_value_cols.append(normalName)

    series = TimeSeries.from_dataframe(
        data_frame, value_cols=targate_value_cols)
    if len(covariate_value_cols) > 0:
        covariate_series = TimeSeries.from_dataframe(
            data_frame, value_cols=covariate_value_cols)
    else:
        covariate_series = None

    return series, covariate_series


def build_model(model_type, target_series, covariate_series, val_series, val_covariate_series, load=False):
    print('Model Type is ', model_type)

    if model_type == 'arima':
        from arima.model import create_model, load_model, fit_model
    elif model_type == 'nbeats':
        from nbeats.model import create_model, load_model, fit_model
    elif model_type == 'nhits':
        from nhits.model import create_model, load_model, fit_model
    elif model_type == 'blockrnn':
        from blockrnn.model import create_model, load_model, fit_model
    elif model_type == 'tcn':
        from tcn.model import create_model, load_model, fit_model
    elif model_type == 'tft':
        from tft.model import create_model, load_model, fit_model
    # elif model_type == 'transformer':
    #     from transformer.model import buildModel, getBestModel, getBestModelForTuning
    # elif model_type == 'rnn':
    #     from rnn.model import buildModel
    # elif model_type == 'lightGBM':
    #     from lightGBM.model import buildModel, getBestModel, getBestModelForTuning
    # elif model_type == 'catBoost':
    #     from catBoost.model import buildModel, getBestModel, getBestModelForTuning
    # elif model_type == 'XGBoost':
    #     from XGBoost.model import buildModel, getBestModel, getBestModelForTuning

    else:
        print('Invalid model name supplied ')
        raise
    if load == 'True':
        model = load_model()
    else:
        model = create_model()
        fit_model(model, target_series, covariate_series,
                  val_series, val_covariate_series)

    return model


def plot_predicted(time_series, plot_labels, plot_colors, plot_file="./figures/actual_vs_predicted.png"):
    i = 0
    for series in time_series:
        plot_array = series.values()
        plt.plot(plot_array, color=plot_colors[i], label=plot_labels[i])
        i += 1
    plt.legend()
    plt.savefig(plot_file)
    plt.show()


def get_metrics(inputs, targets):
    inputs_arr = inputs.values()
    targets_arr = targets.values()

    # mse
    mse_loss = nn.MSELoss(reduction='mean')
    loss = mse_loss(torch.tensor(inputs_arr.astype(float)),
                    torch.tensor(targets_arr.astype(float)))

    # r2_score
    r2 = r2_score(targets_arr, inputs_arr)

    # mae
    mae_loss = nn.L1Loss()
    mae = mae_loss(torch.tensor(targets_arr.astype(float)),
                   torch.tensor(inputs_arr.astype(float)))

    return loss.item(), r2, mae.item()


def predict_for_all_data(start_series, series_to_predict, model, forecast_period, input_lag, start_covariate_series=None, covariate_series=None, model_type='arima'):
    values_to_forecast = len(series_to_predict)
    no_of_iterations = int(values_to_forecast/forecast_period) + \
        (values_to_forecast % forecast_period)

    start_position = 0
    prediction_list = []
    combined_series = start_series[-input_lag:].concatenate(
        series_to_predict, ignore_time_axis=True)

    # Set combined covariate if past covariate is required

    if covariate_series is None:
        combined_covariate = None
    else:
        combined_covariate = start_covariate_series[-input_lag:].concatenate(
            covariate_series, ignore_time_axis=True)

    print("Number of iterations ", no_of_iterations,
          "Target length ", values_to_forecast)

    for i in range(no_of_iterations):
        start_position = forecast_period * i

   #  Set past covariate based on combined covariate if it exists (Not none)
        past_covariate = None
        if combined_covariate is not None:
            past_covariate = combined_covariate[:input_lag+start_position]

            prediction_series = predict_for_model(model_type, model,
                                                  forecast_period, series=combined_series[:input_lag +
                                                                                          start_position],
                                                  past_covariates=past_covariate)

        prediction_list.append(list(prediction_series.values().flatten()))
        # print("interim predict values ", prediction_list)
    prediction_list = reduce(concat, prediction_list)
   # print("predict values for iteration ", i, " Start position ",
   #       start_position, "are \n", prediction_list)

    prediction_list = prediction_list[:values_to_forecast]

    return prediction_list


def predict_for_model(model_type, model, forecast_period, series,
                      past_covariates=None):
    if model_type == 'arima':
        from arima.model import predict_model
    elif model_type == 'nbeats':
        from nbeats.model import predict_model
    elif model_type == 'nhits':
        from nhits.model import predict_model
    elif model_type == 'blockRNN':
        from blockrnn.model import predict_model
    elif model_type == 'tcn':
        from tcn.model import predict_model
    elif model_type == 'tft':
        from tft.model import predict_model

    return predict_model(model, forecast_period, series,
                         past_covariates=None)
