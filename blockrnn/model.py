from darts.models import BlockRNNModel

from pytorch_lightning.callbacks import EarlyStopping

import torch

use_cuda = torch.cuda.is_available()
accelerator = "gpu" if use_cuda else "cpu"
devices = [0] if use_cuda else 1


def default_model_config():

    early_stopper = EarlyStopping(
        "val_loss", min_delta=0.00005, patience=4, verbose=True)
    callbacks = [early_stopper]

    model_config = {'input_chunk_length': 64,
                    'output_chunk_length': 3,
                    # 'generic_architecture': True,
                    # 'num_stacks': 4,
                    # 'num_blocks': 2,
                    # 'num_layers': 4,
                    # 'layer_widths': 512,
                    'n_epochs': 50,
                    'nr_epochs_val_period': 1,
                    'batch_size': 64,
                    'optimizer_kwargs': {'lr': 0.0001},
                    'pl_trainer_kwargs': {
                        'accelerator': accelerator,
                        'devices': devices,
                        'callbacks': callbacks},
                    'save_checkpoints': True,
                    'force_reset': True,
                    'model_name': 'nbeats_run'}
    return model_config


def create_model():
    model_config = default_model_config()
    model = BlockRNNModel(**model_config)
    return model


def load_model():
    model = BlockRNNModel.load("./blockrnn/model.pkl")
    return model


def fit_model(model, target_series, covariate_series=None, val_series=None, val_covariate_series=None):
    model.fit(series=target_series, val_series=val_series,
              past_covariates=covariate_series, val_past_covariates=val_covariate_series, verbose=True,)
    model.save("./blockrnn/model.pkl")
    return


def predict_model(model, forecast_period, series,
                  past_covariates=None):
    prediction_series = model.predict(forecast_period, series, past_covariates)
    return prediction_series
