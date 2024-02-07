# from samecode.plot.pyplot import subplots
# from samecode.survival.plot import KMPlot
# from samecode.logger.mlflow import Logger
import yaml

from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns

from PBMF.attention.model_zoo.SimpleModel import Net
from PBMF.attention.model_zoo.Ensemble import EnsemblePBMF
from samecode.random import set_seed

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train(kwargs):

    args = yaml.safe_load(
        open(kwargs.config, 'r')
    )

    # Check if the directory exists: 
    if os.path.isdir(args['logger']['run_path']):
        if not kwargs.force:
            print('Directory: {} exists, use --force to rewrite content'.format(args['logger']['run_path']))
            exit()
        else:
            os.system('rm -r {}'.format(args['logger']['run_path']))
   
    time = args['outcomes']['time']
    event = args['outcomes']['event']
    treatment = args['outcomes']['treatment']
    control = args['outcomes']['control']
    arms = args['outcomes']['arm_column']
    seed = args['pbmf']['seed']

    # Data
    data = pd.read_csv(args['data']['file'], sep=args['data']['separator'])
    data = data[data[arms].isin([treatment, control])]
    data['treatment'] = data[arms].replace([control, treatment], [0, 1])

    # Features
    ft=[]
    for fi in args['features'].values():
        ft+=fi

    # print(data[arms].value_counts())
    for data_split in range(args['pbmf']['replicas']):

        artifacts = "{}/{}_data_split".format(args['logger']['run_path'], data_split)

        data_train, data_test = train_test_split(data, test_size=args['pbmf']['test_size'], random_state=data_split, stratify=data[[arms, event]])
        data_train = data_train.reset_index(drop=True).copy()
        data_test = data_test.reset_index(drop=True).copy()

        params = dict(
            ignore_patients_frac=args['pbmf']['ignore_patients_frac'], # During training, ignore this % of patients before computing the loss
            layers=args['pbmf']['layers'],
            epochs=args['pbmf']['epochs'],
            minp=args['pbmf']['minp'], # force the model to get this minimum population
            w1=args['pbmf']['w1'], 
            w2=args['pbmf']['w2'],
            seed=seed,
            # embeddings_dim=args['pbmf']['embeddings_dim'],
            learning_rate=args['pbmf']['learning_rate'],
            shuffle=args['pbmf']['shuffle'],
            shuffle_features=args['pbmf']['shuffle_features'],
            l1=args['pbmf']['l1'],
        )

        pbmf = EnsemblePBMF(
            time=time, 
            event=event,
            treatment='treatment',
            stratify='treatment', # used for discarding patients on each model
            features = ft,
            discard_n_features=args['pbmf']['discard_n_features'],
            architecture=Net, 
            **params
        )

        pbmf.fit(
            data_train, 
            num_models=args['pbmf']['num_models'],
            n_jobs=args['pbmf']['n_jobs'],
            test_size=args['pbmf']['ignore_samples_frac'], #Fraction of random patients being ignored from training for each model
            outdir=artifacts,
            save_freq=args['pbmf']['save_freq'],
        )

        # data_train.to_csv("{}/data_train.csv".format(artifacts))
        # data_test.to_csv("{}/data_test.csv".format(artifacts))

        # # Model Performance
        # pbmf = EnsemblePBMF()
        # pbmf.load(
        #     architecture=Net,
        #     outdir=artifacts,
        #     num_models=args['pbmf']['num_models'],
        # )

        # data_train['consensus_risk'], risks = pbmf.parallel_predict(data_train, epoch=args['pbmf']['epochs'], n_jobs=args['pbmf']['n_jobs'], return_all_predictions=True)
        # data_train = pd.concat([data_train, risks], axis = 1)
        # data_test['consensus_risk'], risks = pbmf.parallel_predict(data_test, epoch=args['pbmf']['epochs'], n_jobs=args['pbmf']['n_jobs'], return_all_predictions=True)
        # data_test = pd.concat([data_test, risks], axis = 1)

        # data_train['bin_risk'] = (data_train['consensus_risk'] > 0.5).replace([False, True], ['B-', 'B+'])
        # data_test['bin_risk'] = (data_test['consensus_risk'] > 0.5).replace([False, True], ['B-', 'B+'])

        # data_train.to_csv("{}/data_train.csv".format(artifacts))
        # data_test.to_csv("{}/data_test.csv".format(artifacts))

