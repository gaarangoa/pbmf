import tensorflow as tf 
import numpy as np 
from samecode.random import set_seed

class DataIterator(tf.keras.utils.Sequence):
    
    def __init__(self, **kwargs):
        set_seed(kwargs.get('seed', 0))
        self.X = kwargs.get('X', [])                
        self.y = np.array(kwargs.get('y', []))   
        
        self.features = kwargs.get('features', [])  
        self.batch_size = kwargs.get('batch_size', self.X.shape[0])
        self.shuffle = kwargs.get('shuffle', False)
        self.ignore = kwargs.get('ignore', 0.05) # ignore this fraction of the patients. 
        self.shuffle_features = kwargs.get('shuffle_features', False)
        
        self.on_epoch_end()

    def __len__(self, ):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.X.shape[0] / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        indexes = indexes[:int(np.floor(len(indexes)*(1-self.ignore)))]
        
        X = self.X[indexes]
        try:
            y = self.y[indexes]
        except:
            # During inference no need of y
            y = np.array([])
            
        idx = np.array(range(X.shape[1]))
        idxs = np.zeros( (X.shape[0], X.shape[1] + 1) )
        Xp = np.zeros( (X.shape[0], X.shape[1] + 1) )
        
        flen = len(self.features)
        
        for ix, row in enumerate(X):
            if self.shuffle_features:
                np.random.shuffle(idx)
            
            idxs[ix] = np.concatenate([idx.copy(), [flen]])
            Xp[ix] = np.concatenate([row[idx].copy(), [1]])
        
        return [Xp[..., np.newaxis], idxs], y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)