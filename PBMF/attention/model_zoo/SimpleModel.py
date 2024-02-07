import tensorflow as tf 
from PBMF.attention.loss import  PredLoss
from PBMF.attention.loss import  biomarker_index
from PBMF.attention.metrics import *
from samecode.random import set_seed

class SimpleModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.activation = kwargs.get('activation', 'relu')
        self.out_layers = kwargs.get('out_layers', [32])
        self.num_features = kwargs.get('num_features', 0)

        l1 = kwargs.get('l1', 0.0)
        l2 = kwargs.get('l2', 0.0)
        
        self.predictor = []
        for layer, units in enumerate(self.out_layers):
            self.predictor.append(
                tf.keras.layers.Dense(units, activation=self.activation, name='PredictorLayer{}'.format(layer), kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
            )

        self.score = tf.keras.layers.Dense(2, use_bias=False, activation='softmax', name='Scorer')
    
    def call(self, inp, training=False):
        values = inp[0]
        names = inp[1]
        
        # The data generator adds an artificial feature that we need to ignore here. That feature is always 1 and it is used for attention-based models 
        context = values[:, :-1, 0]
        for predictor_layer in self.predictor:
            context = predictor_layer(context)
        
        score = self.score(context)
        
        return score, context, values, score
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        X, y = data
        
        with tf.GradientTape() as tape:
            pred = self(X, training=True)  # Forward pass
            loss = self.compiled_loss(y, pred[0])
            loss += tf.add_n(self.losses)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, pred[0])
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics} 

def Net(**kwargs):
    '''
    Input Variables:
    **kwargs: a dictionary of hyperparameters to be passed to the model. The following keys are supported:
        seed: the random seed to use for the model (default: 0)
        learning_rate: the learning rate for the optimizer (default: 0.01)
        l1: the L1 regularization strength (default: 0.0)
        l2: the L2 regularization strength (default: 0.0)
        activation: the activation function to use for the model (default: 'relu')
        dropout: the dropout rate for the model (default: 0)
        layers: a list of integers representing the number of hidden layers in the model (default: [32])
        dim: the dimensionality of the input data (default: 32)
        minp: the minimum proportion of samples to be used for early stopping (default: 0.5)
        w1, w2, w3, and w4: the weights for the loss parameters (default: [1, 1, 1, 1])
        num_features: the number of features in the input data (default: 0)

    Output Variables:
        model: a Keras model object representing the neural network.
        loss: the loss function to use for training the model.
        opt: an optimizer object to be used for training the model.
        dummy_data: a list of two dummy data points, used for testing purposes.

    '''


    set_seed(kwargs.get('seed', 0))
    
    lr = kwargs.get('learning_rate', 0.01)
    l1 = kwargs.get('l1', 0.0)
    l2 = kwargs.get('l2', 0.0)
    activation = kwargs.get('activation', 'relu')
    dropout = kwargs.get('dropout', 0)
    layers = kwargs.get('layers', [32])
    dim = kwargs.get('dim', 32)
    minp = kwargs.get('minp', 0.5)
    w1 = kwargs.get('w1', 1)
    w2 = kwargs.get('w2', 1)
    w3 = kwargs.get('w3', 1)
    w4 = kwargs.get('w4', 1)
    
    Loss = kwargs.get('loss', PredLoss)
    loss=PredLoss(minp=minp, w4=w4, w1=w1, w2=w2, w3=w3)
    
    num_features = kwargs.get('num_features', 0)
    
    model = SimpleModel(
        activation=activation, 
        dff=num_features,  
        dim=dim, out_layers=layers, 
        num_features=num_features, 
        l1=l1, l2=l2
    )

    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=[biomarker_index, logrank_hazard_ratio_treatment, logrank_hazard_ratio_control])

    model.__dummy_data__ = [tf.zeros((1, num_features+1, 1)), tf.zeros((1, num_features+1))]
    
    return model
