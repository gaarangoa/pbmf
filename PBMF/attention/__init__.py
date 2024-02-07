import os
import tensorflow as tf
import yaml
from .model import Net
from .dataloader import DataIterator
from ..dataset import DataLoader
import pickle
import numpy as np 
import pandas as pd
from samecode.logger.logger import logger

log = logger('PBMF')

def make_param_file(**kwargs):
    """
    Creates a yaml file with the specified options to be run with the PBMF model. 
    This modeling depends on the  

    Args: 
        time (str): time variable name 
        event (str): event variable name
        tretment (str): treatment variable (0 control, 1 treatment) name
        features: List of features used to train the model. 

        output: directory where to save the model

    Usage: 
        time = 'time'
        event = 'death'
        treatment = 'hormone_therapy'

        features = [
            'age',
            'meno',
            'size',
            'grade',
            'nodes',
            'pr',
            'er',
        ]

        make_param_file(
            time, event, treatment, features,
            ignore_patients_frac=0.1, # During training, ignore this % of patients before computing the loss
            layers=[64],
            epochs=10000,
            minp=0.1, # force the model to get this minimum population
            w1=1.0, 
            w2=1.0,
        )
    """
    time = kwargs.get('time', None)
    event = kwargs.get('event', None)
    treatment = kwargs.get('treatment', None)
    features = kwargs.get('features', None)

    outdir = kwargs.get('outdir', './pbmf_model/')

    os.system('mkdir -p {outdir}'.format(outdir=outdir))
    os.system('rm -rf {outdir}/*'.format(outdir=outdir))

    # data parameters
    seed=kwargs.get('seed', 0)

    dataloader_params = dict(
        features=features,              # List of features used to train the model
        time=time,      # Column name with survival time
        event=event,      # Column name with survival event
        treatment=treatment, # Column name with treatment variables 0: reference treatment, 1: target treatment.
        normalize=kwargs.get('normalize', 'minmax'),         # Normalize the data
        omega_round=kwargs.get('omega_round', 1)
    )

    train_dataiterator_params = dict(
        shuffle=kwargs.get('shuffle', True), 
        seed=seed,
        shuffle_features=kwargs.get('shuffle_features', True),
        ignore=kwargs.get('ignore_patients_frac', 0.1)

    )

    # Model parameters
    layers = kwargs.get('layers', [64]) # attention hidden layer, 
    lr = kwargs.get('learning_rate', 0.001)
    epochs = kwargs.get('epochs', 10000)

    model_params = dict(
        layers=layers,
        seed=seed, 
        learning_rate=lr,
        activation='relu',
        num_features=len(features),
        dim=kwargs.get('embeddings_dim', 32),
        minp=kwargs.get('minp', 0.1), 
        w1=kwargs.get('w1', 1.0),
        w2=kwargs.get('w2', 1.0),
        w3=kwargs.get('w3', 0.0),
        w4=kwargs.get('w4', 0.0),
        l1=kwargs.get('l1', 0.1),
        epochs=kwargs.get('epochs', 10000),
    )

    yaml.dump(
        dict(
            dataloader_params=dataloader_params,
            train_dataiterator_params=train_dataiterator_params,
            model_params=model_params,
            training_params={'epochs':epochs}
        ),
        open('{}/parameters.yaml'.format(outdir), 'w'),
    )

    #logger.info('{}/parameters.yaml'.format(outdir))

    return model_params, dataloader_params, train_dataiterator_params

class PBMF():
    def __init__(self, **kwargs):
        self.Network = kwargs.get('architecture', Net)
        self.DataLoader = kwargs.get('DataLoader', DataLoader)
        self.DataIterator = kwargs.get('DataIterator', DataIterator)
    
    def set_parameters(self, **kwargs):
        # self.DataLoader = kwargs.get('DataLoader', None)
        # self.DataIterator = kwargs.get('DataIterator', None)
        self.artifacts = kwargs.get('outdir', './run/')
        self.features = kwargs.get('features', None)
        
        params = make_param_file( 
            **kwargs
        )

        self.get_params()
        
    def get_params(self):
        self.params = yaml.safe_load(open('{}/parameters.yaml'.format(self.artifacts)))
        self.dataloader_params = self.params['dataloader_params']
        self.train_dataiterator_params = self.params['train_dataiterator_params']
        self.model_params = self.params['model_params']
    
    def get_biomarker_index(self, epoch=None):
        metrics  = pd.read_csv('{}/metrics.csv'.format(self.artifacts))
        
        if epoch == None:
            bix = metrics.iloc[-1].biomarker_index
        else:
            bix = metrics.iloc[epoch-1].biomarker_index
            # print(metrics.iloc[epoch-1])

        return bix
    
    def set_data(self, data, **kwargs):
        
        self.dataloader = self.DataLoader(
         **self.dataloader_params
        )

        X_train, y_train = self.dataloader.fit(data.fillna(0))
        if kwargs.get('normalize_lambda', False): 
            y_train['lambdas'] = y_train['lambdas'] / np.max(y_train['lambdas'])

        # save the dataloader for future runs.
        pickle.dump(self.dataloader, open('{}/dataloader.pk'.format(self.artifacts), 'wb'))

        self.train_iterator = self.DataIterator(
            X=X_train, 
            y=y_train, 
            features=self.features, 
            **self.train_dataiterator_params
        )
    
    def load_model(self, artifacts, model, **kwargs):
        self.artifacts = artifacts
        self.get_params()
        self.dataloader = pickle.load(open('{}/dataloader.pk'.format(self.artifacts), 'rb'))
        self.features = self.dataloader_params['features']

        self.model = self.model = self.Network(
            **self.model_params
        )

        self.model.predict(self.model.__dummy_data__);
        self.model.load_weights('{}/{}'.format(self.artifacts, model))
        self.bix = self.get_biomarker_index(epoch=int(model.replace('cp-', '').replace('.h5', '')))

    def predict(self, data, **kwargs):

        # we don't want to shuffle patients or features, nor ignore patients when making predictions
        data_iterator_params = self.train_dataiterator_params
        data_iterator_params['ignore'] = kwargs.get('ignore', 0.0)
        data_iterator_params['shuffle'] = kwargs.get('shuffle', False)
        data_iterator_params['shuffle_features'] = kwargs.get('shuffle_features', False)
        
        
        X, y = self.dataloader.transform(data)
        X_iterator = self.DataIterator(
            X=X, 
            y=y,
            features = self.features,
            **data_iterator_params
        )

        return self.model.predict(X_iterator)

    def fit(self, **kwargs):
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.artifacts+'/cp-{epoch:09d}.h5',
            monitor=kwargs.get('monitor', 'loss'),
            save_freq=kwargs.get('save_freq', 1000),
            save_weights_only=kwargs.get('save_weights_only', True),
            verbose=0
        )
    
        self.model = self.Network(
            **self.model_params
        )

        # Model training
        self.history = self.model.fit(
            self.train_iterator,
            epochs=self.params['model_params']['epochs'],
            verbose=0,
            callbacks=[model_checkpoint]+kwargs.get('callbacks', []),
        )

        metrics = pd.DataFrame(self.history.history)
        metrics.to_csv('{}/metrics.csv'.format(self.artifacts), index=False)