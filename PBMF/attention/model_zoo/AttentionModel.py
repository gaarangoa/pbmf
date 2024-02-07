import tensorflow as tf 
from samecode.random import set_seed
from PBMF.attention.loss import  PredLoss
from PBMF.attention.loss import  biomarker_index
from PBMF.attention.metrics import *

class AttentionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs.get('dim', 2)
        self.heads = kwargs.get('heads', 2)
        self.dff = kwargs.get('dff', 0)
        self.activation = kwargs.get('activation', 'relu')
        self.out_layers = kwargs.get('out_layers', [32])
        self.num_features = kwargs.get('num_features', 0)

        l1 = kwargs.get('l1', 0.0)
        l2 = kwargs.get('l2', 0.0)
        
        self.embeddings = tf.keras.layers.Embedding(self.dff+1, self.dim, mask_zero=False, name='NameEmbeddings')
        self.numerical_embeddings = tf.keras.layers.Dense(self.dim, name='NumericalEmbeddings')
        
        self.query = tf.keras.layers.Dense(self.dim, name='Query', kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2))
        self.value = tf.keras.layers.Dense(self.dim, name='Value', kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2))
        
        self.attention = tf.keras.layers.AdditiveAttention(name='Attention')
        
        self.predictor = []
        for layer, units in enumerate(self.out_layers):
            self.predictor.append([
                tf.keras.layers.Dense(units, activation=self.activation, name='PredictorLayer{}'.format(layer), kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)),
                tf.keras.layers.LayerNormalization(axis=1, name='PredictorNorm{}'.format(layer))
            ])

        self.score = tf.keras.layers.Dense(2, use_bias=False, activation='softmax', name='Scorer', kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2))
    
    def call(self, inp, training=False):
        values = inp[0]
        names = inp[1]
        
        v = self.numerical_embeddings(values)
        e = self.embeddings(names)
        
        mix_embeddings = e+v
        
        # tf.reduce_mean(mix_embeddings, axis=1, keepdims=True)
        query = self.query(mix_embeddings[:, -1:, :])
        value = self.value(mix_embeddings[:, :-1, :])
        embeddings, attentions = self.attention([query, value], training=training, return_attention_scores=True)
        
        context = tf.reduce_mean(embeddings + mix_embeddings, axis=1)
        
        for predictor_layer, norm_layer in self.predictor:
            context = predictor_layer(context)
            context = norm_layer(context)
        
        # attention = self.out_attention(context)

        # map the values back to the input
        # out_values = tf.math.multiply(values[:, :-1, 0], attention, name='OutValues')
        score = self.score(context)
        
        return score, embeddings, attentions, context
    
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

    loss=PredLoss(minp=minp, w4=w4, w1=w1, w2=w2, w3=w3)
    
    num_features = kwargs.get('num_features', 0)
    
    model = AttentionModel(
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
