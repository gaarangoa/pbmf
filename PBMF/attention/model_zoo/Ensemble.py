from PBMF.attention import PBMF
from samecode.random import set_seed
from samecode.logger.logger import logger
import numpy as np 
from sklearn.model_selection import train_test_split
import multiprocessing
from tqdm.auto import tqdm
import pandas as pd
import tensorflow as tf

log = logger('PBMF')

class EnsemblePBMF():
    def __init__(self, **kwargs):
        '''        
        USAGE: 
        
        seed = 0
            params = dict(
                ignore_patients_frac=0.1, # During training, ignore this % of patients before computing the loss
                layers=[64],
                epochs=10,
                minp=0.5, # force the model to get this minimum population
                w1=1.0, 
                w2=0.0,
                seed=0,
                embeddings_dim=32,
                learning_rate=0.01,
                shuffle=True,
                shuffle_features=False,
                l1=0.0,
            )
        
        pbmf = EnsemblePBMF(
            time=time, 
            event=event,
            treatment=treatment,
            stratify=treatment,
            features = features,
            discard_n_features=2,
            architecture=Net, 
            **params
        )

        pbmf.fit(
            data_train, 
            num_models=2, 
            test_size=0.2, 
            outdir='./runs/',
            save_freq=10,
            metrics=[biomarker_index]
        )
        
        pbmf = EnsemblePBMF()
        pbmf.load(
            architecture=Net,
            outdir='./runs/',
            num_models=2,
        )

        data_train['consensus_risk2'] = pbmf.predict(data_train, epoch=10)

        '''
        
        self.params = kwargs
        self.features = kwargs.get('features', [])
        self.features_random_select = self.features.copy()
        self.discard_n_features = kwargs.get('discard_n_features', 1) 
        self.time = kwargs.get('time', None)
        self.event = kwargs.get('event', None)
        self.treatment = kwargs.get('treatment', None)
        self.stratify = kwargs.get('stratify', None)
        self.architecture = kwargs.get('architecture', None)
        
    def random_feature_sets(self, num_models):
        set_seed(self.params.get('seed', None))
        feature_sets = []
        for i in range(num_models):
            np.random.shuffle(self.features_random_select)
            if self.discard_n_features == 0:
                feature_sets.append(self.features_random_select)
            else:
                feature_sets.append(self.features_random_select[:-self.discard_n_features])
            
        return feature_sets
    
    def fit_one(self, data_train, test_size=0.2, seed=None, outdir='./runs/', save_freq=100, normalize_lambda=False, **kwargs):
        set_seed(self.params.get('seed', None)) # this is the model seed. It is universal, all models have the same seed. 
        tf.keras.backend.clear_session()
        # The seed below indicates the different split.
        data_train_, _ = train_test_split(data_train, test_size=test_size, random_state=seed, stratify=data_train[self.stratify])
        data_train_ = data_train_.reset_index(drop=True).copy()
        
        self.params_seed = self.params.copy()
        self.params_seed['features'] = self.feature_sets[seed]

        pbmf = PBMF(architecture=self.architecture)
        pbmf.set_parameters(
            outdir='{}/{}/'.format(outdir, seed),
            **self.params_seed
        )
        
        pbmf.set_data(data_train_, normalize_lambda=normalize_lambda)
        pbmf.fit(save_freq=save_freq)

        return True
        
    
    def fit(self, data_train, num_models=10, test_size=0.2, outdir='./runs/', save_freq=100, normalize_lambda=False, **kwargs):
        
        self.feature_sets = self.random_feature_sets(num_models)
        self.outdir = outdir
        self.num_models = num_models
        n_jobs = kwargs.get('n_jobs', False)

        if not n_jobs or n_jobs == 1:         
            for seed in range(num_models):
                self.fit_one(
                    data_train,
                    test_size=test_size,
                    seed=seed,
                    outdir=outdir,
                    save_freq=save_freq,
                    normalize_lambda=normalize_lambda,
                    **kwargs
                )
        else: 
            arg_iterable = [(data_train, test_size, seed, outdir, save_freq, normalize_lambda) for seed in range(num_models)]
            with multiprocessing.Pool(n_jobs) as pool:
                res = pool.starmap(self.fit_one, arg_iterable)

            pool.close()
            pool.join()
    
    def load(self, **kwargs):
        self.num_models = kwargs.get('num_models', None)
        self.architecture = kwargs.get('architecture', None)
        self.outdir = kwargs.get('outdir', None)
        
    
    def predict(self, data, epoch, **kwargs):
        '''
        data: dataframe with the input data to predict.
        epoch: the specific epoch to load to make the predictions.
        return_all_models_predictions: Adds to input data frame all the individual models predictions (default = False)
        '''
        num_models = kwargs.get('num_models', self.num_models)
        architecture = kwargs.get('architecture', self.architecture)
        outdir = kwargs.get('outdir', self.outdir)
        return_all_models_predictions = kwargs.get('return_all_models_predictions', False)

        iterator = range(num_models) if type(num_models) == int else num_models

        consensus = []
        for seed in tqdm(iterator, total=len(iterator)):
            model = PBMF(architecture=architecture)
            model.load_model("{}/{}/".format(outdir, seed), model='cp-{epoch:09d}.h5'.format(epoch=epoch))            
            risk = model.predict(data)[0][:, int(model.bix)]
            if return_all_models_predictions:
                data['risk_seed{}'.format(seed)] = risk
            consensus.append(risk)
            
        return np.mean(consensus, axis=0)


    def load_one(self, model_id, epoch, **kwargs):
        architecture = kwargs.get('architecture', self.architecture)
        outdir = kwargs.get('outdir', self.outdir)

        model = PBMF(architecture=architecture)
        model.load_model("{}/{}/".format(outdir, model_id), model='cp-{epoch:09d}.h5'.format(epoch=epoch))

        return model

    def embeddings(self, model_id, epoch, **kwargs):
        pass

    def predict_one(self, data, epoch, seed):

        #log.info('Processing: START {}'.format(seed))
        tf.keras.backend.clear_session()
        model = PBMF(architecture=self.architecture)
        model.load_model("{}/{}/".format(self.outdir, seed), model='cp-{epoch:09d}.h5'.format(epoch=epoch))
        risk = model.predict(data)[0][:, int(model.bix)]

        #log.info('Processing: END {}'.format(seed))
        
        return risk, seed

    def parallel_predict(self, data, epoch, n_jobs=1, return_all_predictions=False):
        
        iterator_counts = range(self.num_models) if type(self.num_models) == int else self.num_models

        arg_iterable = [(data, epoch, seed) for seed in iterator_counts]
        with multiprocessing.Pool(n_jobs) as pool:
            res = pool.starmap(self.predict_one, arg_iterable)

        pool.close()
        pool.join()
        
        seeds = ["risk_seed{}".format(i[1]) for i in res]
        res = np.array([i[0] for i in res])
        
        if return_all_predictions:
            return res.mean(axis=0), pd.DataFrame(res.T, columns=seeds)
        else:
            return res.mean(axis=0)
        