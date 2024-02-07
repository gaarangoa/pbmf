import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import pandas as pd
import seaborn as sns

from PBMF.attention.model_zoo.SimpleModel import Net
from PBMF.attention.model_zoo.Ensemble import EnsemblePBMF
from PBMF.attention.model_zoo.Pruning import compute_consensus_parallel
from PBMF.attention.model_zoo.Pruning import compute_correlations_parallel
from PBMF.attention.model_zoo.Pruning import select_models

from samecode.plot.pyplot import subplots

import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def distil(kwargs):

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

    thr = args['distil']['thr']
    max_depth = args['distil']['decision_tree']['max_depth']
    decision_tree_seed = args['distil']['decision_tree']['random_state']
    
    # print(data[arms].value_counts())
    for data_split in range(args['pbmf']['replicas']):

        artifacts = "{}/{}_data_split".format(args['logger']['run_path'], data_split)

        data_train = pd.read_csv("{}/data_train-epoch-{}-pruning-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']), low_memory=False)
        data_test = pd.read_csv("{}/data_test-epoch-{}-pruning-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']), low_memory=False)
        
        data_train_ = data_train[~((data_train['pruning_consensus_risk_min'] > 0.5-thr) & (data_train['pruning_consensus_risk_min'] < 0.5+thr))].copy().reset_index(drop=True)
        clf = DecisionTreeClassifier(random_state=decision_tree_seed, max_depth=max_depth)

        clf.fit(
            data_train_[ft], 
            data_train_.bin_risk,
        )

        data_train['tree_predicted_biomarker'] = clf.predict(data_train[ft])
        data_test['tree_predicted_biomarker'] = clf.predict(data_test[ft])   

        data_train.to_csv("{}/data_train-epoch-{}-pruning-{}-distiled.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']), index=False)
        data_test.to_csv("{}/data_test-epoch-{}-pruning-{}-distiled.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']), index=False)     

        with open('{}/tree.pk'.format(artifacts), 'wb') as f:
            pickle.dump(clf, f)

        # f, axs = subplots(cols=1, rows=1, w=5, h=5, return_f=True)
        # text_tree = tree.plot_tree(clf, 
        #            feature_names=ft,  
        #            class_names=['B+', 'B-'],
        #            filled=True, ax=axs[0]
        # )

        # f.savefig('{}/decision_tree.pdf'.format(artifacts))