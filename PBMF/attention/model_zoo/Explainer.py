import networkx as nx
import tensorflow as tf
import pandas as pd
import numpy as np 


from sklearn import metrics
from sklearn.inspection import permutation_importance
from lifelines.utils import concordance_index as cindex
from lifelines import CoxPHFitter

def Grad4Input(pbmf, data, layer=None, **kwargs):
    # Convert the input data to a TensorFlow tensor
    data_iterator_params = pbmf.train_dataiterator_params
    data_iterator_params['ignore'] = kwargs.get('ignore', 0.0)
    data_iterator_params['shuffle'] = kwargs.get('shuffle', False)
    data_iterator_params['shuffle_features'] = kwargs.get('shuffle_features', False)
    
    return_all_gradients = kwargs.get('return_all_gradients', False)
    
    model = pbmf.model

    X, y = pbmf.dataloader.transform(data)

    X_iterator = pbmf.DataIterator(
        X=X, 
        y=y,
        features = pbmf.features,
        **data_iterator_params
    )

    X, y = X_iterator.__getitem__(0)
    X0 = tf.Variable(X[0])
    X1 = tf.Variable(X[1])

    with tf.GradientTape() as tape:
        tape.watch(X0)
        pred = model([X0, X1], training=False)
        gradients = tape.gradient(pred[0], X0)
    
    if return_all_gradients:
        gradients = np.array(gradients)
    else:
        gradients = np.mean(gradients, axis=0)

    return gradients


