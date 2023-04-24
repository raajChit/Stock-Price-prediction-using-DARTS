import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from utilities.utilities import get_preprocessed_datasets, get_target_covariate_timeseries, build_model, plot_predicted, get_metrics, predict_for_all_data
import argparse

parser = argparse.ArgumentParser(description='darts algorithms')

parser.add_argument('--model', type=str, default='arima',
                    help='Set Model to use')
parser.add_argument('--search', type=str, choices=['True', 'False'],
                    help='To tune the hyperparameters')
parser.add_argument('--load', type=str, choices=['True', 'False'],
                    help='To create new or load exisitng model')
parser.add_argument('--target', type=str, default='Close',
                    help='Column to predict')
parser.add_argument('--mse', type=str, choices=[
                    'True', 'False'], default='False', help='Execute only to calculate mse')

args = parser.parse_args()
print('Model is ', args.model, ' Search Mode is ', args.search, ' load is ',
      args.load, ' Target Column ', args.target, 'mse value is ', args.mse)


train_set, val_set, test_set = get_preprocessed_datasets()


target_columns = [args.target]
target_series, covariate_series = get_target_covariate_timeseries(
    train_set, target_columns)

val_series, val_covariate_series = get_target_covariate_timeseries(
    val_set, target_columns)

test_series, test_covariate_series = get_target_covariate_timeseries(
    test_set, target_columns)


model = build_model(args.model, target_series, covariate_series,
                    val_series, val_covariate_series, load=args.load)

if args.mse == 'True' and args.model == 'arima':
    forecast_period = [3, 5, 8]
    mse_list = []
    for item in forecast_period:
        predict_value = model.predict(item)
        mse_value, r2_value, mae_value = get_metrics(
            predict_value, val_series[:item])
        mse_list.append(mse_value)
        print("\nmse value for forecast period ", item, " is : ", mse_value,
              "\nr2 value for forecast period ", item, " is : ", r2_value, "\nmae value for forecast period ", item, " is : ", mae_value)
    plt.xlabel('Forecast period')
    plt.ylabel('MSE values')
    plt.plot(forecast_period, mse_list)
    plt.savefig('MSE_by_forecast_period.png')
    plt.show()

if args.mse == 'True' and args.model != 'arima':
    predicted_array = np.array(predict_for_all_data(
        target_series, val_series, model, 3, 64, start_covariate_series=covariate_series, covariate_series=val_covariate_series, model_type=args.model)).flatten()
    predicted_series = TimeSeries.from_values(predicted_array)
    mse_value, r2_value, mae_value = get_metrics(predicted_series, val_series)
    print("mse value is :", mse_value)
    print("mae value is :", mae_value)
    print("r2 score is :", r2_value)

else:
    predicted_array = np.array(predict_for_all_data(
        target_series, val_series, model, 3, 64, start_covariate_series=covariate_series, covariate_series=val_covariate_series, model_type=args.model)).flatten()

    plot_labels = ["Actual Values", "Predicted Values"]
    plot_colors = ['r', 'b', 'k']

    predicted_series = TimeSeries.from_values(predicted_array)

    plot_predicted([val_series, predicted_series], plot_labels, plot_colors,
                   plot_file=f"./figures/{args.model}_actual_vs_predicted.png")

#    mse_value, r2_value = get_metrics(predicted_series, val_series)
#    print("mse value is : ", mse_value, "\nr2 value is : ", r2_value)
exit(0)

mse_list = []
r2_list = []
for i in range(1, 16):
    predict_val = model.predict(i)
    mse_item, r2_item = get_metrics(predict_val, val_series[:i])
    mse_list.append(mse_item)
    r2_list.append(r2_item)


plot_predicted([val_series[:15], predict_val], plot_labels, plot_colors)

plt.plot(mse_list)
plt.savefig('mse_list.png')
plt.show()
plt.plot(r2_list)
plt.savefig('r2_list.png')
plt.show()
