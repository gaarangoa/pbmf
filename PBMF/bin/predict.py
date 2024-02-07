# from samecode.plot.pyplot import subplots
# from samecode.survival.plot import KMPlot
# from samecode.logger.mlflow import Logger
import yaml

from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns

from PBMF.attention.model_zoo.SimpleModel import Net
from PBMF.attention.model_zoo.Ensemble import EnsemblePBMF

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def predict(kwargs):

    args = yaml.safe_load(
        open(kwargs.config, 'r')
    )
   
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
        data_train = data_train.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)

        # data_train = pd.read_csv("{}/data_train.csv".format(artifacts))
        # data_test = pd.read_csv("{}/data_test.csv".format(artifacts))

        # Model Performance
        pbmf = EnsemblePBMF()
        pbmf.load(
            architecture=Net,
            outdir=artifacts,
            num_models=args['pbmf']['num_models'],
        )

        data_train['consensus_risk'], risks = pbmf.parallel_predict(data_train, epoch=args['results']['epoch'], n_jobs=args['pbmf']['n_jobs'], return_all_predictions=True)
        data_train = pd.concat([data_train, risks], axis = 1)
        data_test['consensus_risk'], risks = pbmf.parallel_predict(data_test, epoch=args['results']['epoch'], n_jobs=args['pbmf']['n_jobs'], return_all_predictions=True)
        data_test = pd.concat([data_test, risks], axis = 1)

        data_train['bin_risk'] = (data_train['consensus_risk'] > 0.5).replace([False, True], ['B-', 'B+'])
        data_test['bin_risk'] = (data_test['consensus_risk'] > 0.5).replace([False, True], ['B-', 'B+'])

        data_train.to_csv("{}/data_train-epoch-{}.csv".format(artifacts, args['results']['epoch']))
        data_test.to_csv("{}/data_test-epoch-{}.csv".format(artifacts, args['results']['epoch']))