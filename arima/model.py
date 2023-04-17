from darts.models import ARIMA


def create_model():
    model = ARIMA(p=64, d=4, q=6)
    return model


def load_model():
    model = ARIMA.load("./arima/model.pkl")
    return model


def fit_model(model, target_series, covariate_series=None, val_series=None, val_covariate_series=None):
    model.fit(target_series)
    model.save("./arima/model.pkl")
    return


def predict_model(model, forecast_period, series,
                  past_covariates=None):
    prediction_series = model.predict(forecast_period, series)
    return prediction_series
