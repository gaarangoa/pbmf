# from samecode.plot.pyplot import subplots
# from samecode.survival.plot import KMPlot
# from samecode.logger.mlflow import Logger
import yaml

from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns

from PBMF.attention.model_zoo.SimpleModel import Net
from PBMF.attention.model_zoo.Ensemble import EnsemblePBMF
from PBMF.attention.model_zoo.Pruning import compute_consensus_parallel
from PBMF.attention.model_zoo.Pruning import compute_correlations_parallel
from PBMF.attention.model_zoo.Pruning import select_models

from samecode.random import set_seed

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def pruning(kwargs):

    args = yaml.safe_load(
        open(kwargs.config, 'r')
    )
   
    time = args['outcomes']['time']
    event = args['outcomes']['event']
    treatment = args['outcomes']['treatment']
    control = args['outcomes']['control']
    arms = args['outcomes']['arm_column']
    seed = args['pbmf']['seed']

    # Features
    ft=[]
    for fi in args['features'].values():
        ft+=fi

    # print(data[arms].value_counts())
    for data_split in range(args['pbmf']['replicas']):

        artifacts = "{}/{}_data_split".format(args['logger']['run_path'], data_split)

        data_train = pd.read_csv("{}/data_train-epoch-{}.csv".format(artifacts, args['results']['epoch']), low_memory=False)
        data_test = pd.read_csv("{}/data_test-epoch-{}.csv".format(artifacts, args['results']['epoch']), low_memory=False)

        # COMPUTE CONSENSUS BASED ON MUTUAL RISKS FROM TRAIN AND SELECTED MOST USEFUL MODELS
        res = compute_consensus_parallel(data_train,['risk_seed{}'.format(i) for i in range(args['pbmf']['num_models'])], thr = 0.5, n_jobs = args['pbmf']['n_jobs'])
        res_corr = compute_correlations_parallel(data_train, ['risk_seed{}'.format(i) for i in range(args['pbmf']['num_models'])], res, n_jobs = args['pbmf']['n_jobs'])
        selected_models = select_models(res_corr, drop_negatives = args['pruning']['drop_negatives'], select_percentile = args['pruning']['percentile'])
        
        # RECALCULATE CONSENSUS BASED ON MIN MODEL
        pbmf_min = EnsemblePBMF()
        pbmf_min.load(
            architecture=Net,
            outdir=artifacts,
            num_models = selected_models
        )
        
        data_train['pruning_consensus_risk_min'] = pbmf_min.parallel_predict(data_train, epoch=args['results']['epoch'], n_jobs=args['pbmf']['n_jobs'])
        data_test['pruning_consensus_risk_min'] = pbmf_min.parallel_predict(data_test, epoch=args['results']['epoch'], n_jobs=args['pbmf']['n_jobs']) #parallel_predict

        data_train.to_csv("{}/data_train-epoch-{}-pruning-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']), index=False)
        data_test.to_csv("{}/data_test-epoch-{}-pruning-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']), index=False)