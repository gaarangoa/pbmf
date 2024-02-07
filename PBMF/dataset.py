import numpy as np 
import pandas as pd

from lifelines import KaplanMeierFitter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from .attention.loss import lambdas_

class DataLoader():
    '''
    Generate dataset for training
    
    dataloader = PLFDataset(
        features,
        time='OS_Months',
        event='OS_Event',
        treatment='Treatment',
        normalize=True
    )

    X_train, y_train = dataloader.fit(data_train)
    X_test, y_test = dataloader.transform(data_train)
    '''
    
    
    def __init__(self, features=[], time='time', event='event', treatment='treatment', normalize=True, **kwargs):
        '''This class assumes that the data is already in good shape and with no categorical variables'''

        self.time=time
        self.event=event
        self.lambdas='lambdas'
        self.treatment=treatment
        self.normalize=normalize
        self.features = features
        if self.normalize == True:
            self.scaler = MinMaxScaler()
        if self.normalize == 'minmax': 
            self.scaler = MinMaxScaler()
        if self.normalize == 'zscore':
            self.scaler = StandardScaler()
        self.omega_round=kwargs.get('omega_round', 1)
    
    def fit(self, data, **kwargs):
        
        assert isinstance(data, pd.DataFrame)==True
        
        X = data[self.features].copy()
        y = data[[self.time, self.event, self.treatment]].copy()

        if self.normalize:
            self.scaler.fit(X.fillna(0))
            X = self.scaler.transform(X.fillna(0))
        else:
            X = X.fillna(0).values # if not normalizing, still need to convert to np.array
                
        lambdas=lambdas_(y[self.time], y[self.event], omega_round=self.omega_round) #compute lamdas (see loss function formula for explanation)
        y[self.lambdas]=lambdas
        y = y[[self.time, self.event, self.lambdas, self.treatment]]
        
        
        return X, y
    
    def transform(self, data):
        assert isinstance(data, pd.DataFrame)==True
        
        X = data[self.features].copy()

        if self.normalize:
            X = self.scaler.transform(X.fillna(0))

        try:
            y = data[[self.time, self.event, self.treatment]].copy()        
            lambdas=lambdas_(y[self.time], y[self.event]) #compute lamdas (see loss function formula for explanation)
            y[self.lambdas]=lambdas
            y = y[[self.time, self.event, self.lambdas, self.treatment]]
            
            
            return X, y
        except:
            return X, None