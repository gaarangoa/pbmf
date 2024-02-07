import tensorflow as tf
from .loss import PredictiveBiomarkerLossFunction as PredLoss
from tensorflow.keras import backend as K
from lifelines import CoxPHFitter
from .metrics import MeanTimeMetric

def LinearModel(omega=0.01, seed=0, **kwargs):
    tf.random.set_seed(seed)
    
    lr = kwargs.get('learning_rate', 0.01)
    l1 = kwargs.get('l1', 1e-5)
    l2 = kwargs.get('l2', 1e-5)
    
    loss=PredLoss(omega=omega)
    
    model = tf.keras.Sequential()
    
    model.add(
        tf.keras.layers.Dense(
            1, use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        )
    )
    
    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    
    return model

def AttentionNet(seed=0, omega=0.01, **kwargs):
    tf.random.set_seed(seed)
    
    lr = kwargs.get('learning_rate', 0.01)
    l1 = kwargs.get('l1', 0.0)
    l2 = kwargs.get('l2', 0.0)
    activation = kwargs.get('activation', 'relu')
    dropout = kwargs.get('dropout', 0)
    layers = kwargs.get('layers', [32])
    loss=PredLoss(omega=omega)
    input_size = kwargs.get('input_size', 1)
    feature_shape = kwargs.get('feature_shape', (1, 1))
    weight_params_dim = kwargs.get('weight_params_dim', 32)
    
    
    shared_dense_layer_1 = tf.keras.layers.Dense(layers[0], activation="relu")
    shared_dense_layer_2 = tf.keras.layers.Dense(layers[1], activation="relu")
    shared_norm_1 = tf.keras.layers.LayerNormalization(axis=1)
    shared_norm_2 = tf.keras.layers.LayerNormalization(axis=1)
    
    inp = tf.keras.layers.Input(feature_shape)
    e = shared_dense_layer_1(inp)
    e = shared_norm_1(e)
    e = shared_dense_layer_2(e)
    e = shared_norm_2(e)
    
    alpha = tf.keras.layers.Dense(feature_shape, activation='softmax', name='alpha')(e)    
    multiply = tf.keras.layers.multiply([alpha, inp])
    
    o = tf.keras.layers.Dense(
        layers[2], activation='relu', 
        name='emb2',
    )(multiply)
    
    o = tf.keras.layers.Dropout(dropout)(o)
    
    output = tf.keras.layers.Dense(
        1, use_bias=True, activation='linear',
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)
    )(o)
    
    model = tf.keras.Model([inp], output)

    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    
    return model

def Net(seed=0, omega=0.01, **kwargs):
    tf.random.set_seed(seed)
    
    lr = kwargs.get('learning_rate', 0.01)
    l1 = kwargs.get('l1', 0.0)
    l2 = kwargs.get('l2', 0.0)
    activation = kwargs.get('activation', 'relu')
    dropout = kwargs.get('dropout', 0)
    layers = kwargs.get('layers', [32])
    
    loss=PredLoss(omega=omega)
    
    model = tf.keras.Sequential()
    model.embedding_size = len(layers)

    for ix, dim in enumerate(layers):
        # if dropout > 0:
        #     model.add(
        #         tf.keras.layers.Dropout(dropout)
        #     )

        model.add(
            tf.keras.layers.Dense(
                dim, activation=activation, use_bias=True,
                # kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=seed),
                # kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            )
        )
    
        # model.add(
        #     tf.keras.layers.LayerNormalization(axis=1)
        # )
    
    model.add(
        tf.keras.layers.Dense(
            1, use_bias=False, 
            # kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=seed),
            # kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), 
            activation='linear',
        )
    )

    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    
    return model

class ZScoreNormalizationLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[0])
        z_scores = (inputs - mean) #/ tf.sqrt(variance)
        return z_scores

def OptimNN(hp):

    seed = 0# hp.Int('seed', 0, 10, step=1)
    tf.random.set_seed(seed)

    omega = 1
    loss=PredLoss(omega=omega / 1)

    model = tf.keras.Sequential()

    layers = hp.Int('layers', 1, 3, step=1)
    dropout = hp.Choice('dropout', [0.0, 0.1, 0.2])
    activation = hp.Choice('activation', ['relu'])

    l1 = hp.Choice('l1', [0.1, 0.01, 0.001, 0.0])
    l2 = hp.Choice('l2', [0.1, 0.01, 0.001, 0.0])

    for ix in range(layers):
        if dropout > 0:
            model.add(
                tf.keras.layers.Dropout(dropout)
            )
        
        dim = hp.Choice('dim_{}'.format(ix), [2, 8, 16, 32, 64, 128, 256])
        model.add(
            tf.keras.layers.Dense(
                dim, activation=activation, use_bias=True,
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            )
        )
    
        model.add(
            tf.keras.layers.LayerNormalization(axis=1)
        )
    
    model.add(
        tf.keras.layers.Dense(
            1, use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2), activation='linear',
        )
    )

    lr = hp.Choice('lr', [0.01, 0.001])
    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=[MeanTimeMetric()], run_eagerly=False)
    
    return model

class VirtualTwins(): 
    def __init__(self, model, features, **kwargs):
        '''Implementation in Python from: https://cran.r-project.org/web/packages/aVirtualTwins/vignettes/full-example.html
        
        Model Usage: 

        model_args = dict(
            num_trees = 500,
        )

        model_fit_args = dict(
            max_features="sqrt", max_depth=5, min_node_size=20
        )

        vt = VirtualTwins(
            RandomSurvivalForestModel, 
            features, time=time, event=event, 
            treatment='treatment',
            model_args = model_args,
            model_fit_args = model_fit_args
        )

        vt.fit(data_train)

        data_train['risk'] = vt.predict(data_train)
        data_test['risk'] = vt.predict(data_test)

        '''
        self.time = kwargs.get('time', 'time')
        self.event = kwargs.get('event', 'event')
        self.treatment = kwargs.get('treatment', 'treatment')
        self.model_args = kwargs.get('model_args', {})
        self.model_fit_args = kwargs.get('model_fit_args', {})
        self.features = features
        self.model = model

    def fit(self, df, control=0, treatment=1, **kwargs):
        '''
        Control: control label [default 0]
        Treatment: treatment label [default 1]
        '''

        self.control_model = self.model(**self.model_args)
        self.treatment_model = self.model(**self.model_args)
        
        control_df = df[df[self.treatment] == control]
        treatment_df = df[df[self.treatment] == treatment]
        
        self.control_model.fit(control_df[self.features], control_df[self.time], control_df[self.event], **self.model_fit_args)
        self.treatment_model.fit(treatment_df[self.features], treatment_df[self.time], treatment_df[self.event], **self.model_fit_args)
        
    def predict(self, df):
        control_risk = self.control_model.predict_risk(df[self.features])
        treatment_risk = self.treatment_model.predict_risk(df[self.features])
        
        return treatment_risk - control_risk

class VirtualTwinsOneModel(): 
    def __init__(self, model, features, **kwargs):
        '''Implementation in Python from: https://cran.r-project.org/web/packages/aVirtualTwins/vignettes/full-example.html
        
        Model Usage: 

        model_args = dict(
            num_trees = 500,
        )

        model_fit_args = dict(
            max_features="sqrt", max_depth=5, min_node_size=20
        )

        vt = VirtualTwins(
            RandomSurvivalForestModel, 
            features, time=time, event=event, 
            treatment='treatment',
            model_args = model_args,
            model_fit_args = model_fit_args
        )

        vt.fit(data_train)

        data_train['risk'] = vt.predict(data_train)
        data_test['risk'] = vt.predict(data_test)

        '''
        self.time = kwargs.get('time', 'time')
        self.event = kwargs.get('event', 'event')
        self.treatment = kwargs.get('treatment', 'treatment')
        self.model_args = kwargs.get('model_args', {})
        self.model_fit_args = kwargs.get('model_fit_args', {})
        self.features = features
        self.model = model

    def fit(self, df, control=0, treatment=1, **kwargs):
        '''
        Control: control label [default 0]
        Treatment: treatment label [default 1]
        '''

        self.vt_model = self.model(**self.model_args)
        self.vt_model.fit(df[self.features + [self.treatment]], df[self.time], df[self.event], **self.model_fit_args)
        
    def predict(self, df):
        df['reversed_treatment'] = 0
        control_risk = self.vt_model.predict_risk(df[self.features + ['reversed_treatment']])
        df['control_risk'] = control_risk
        
        df['reversed_treatment'] = 1
        treatment_risk = self.vt_model.predict_risk(df[self.features + ['reversed_treatment']])
        df['treatment_risk'] = treatment_risk
        
        return treatment_risk - control_risk
