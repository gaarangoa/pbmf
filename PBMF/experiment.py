from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from samecode.logger.mlflow import Logger

from PBMF.attention import PBMF
from PBMF.attention.loss import biomarker_index
from PBMF.utils import get_confusion_matrix2, get_AUC, calculate_metrics
import tqdm
from samecode.random import set_seed

from pysurvival.models.survival_forest import RandomSurvivalForestModel
from PBMF.models import VirtualTwins


from PBMF.attention.model_zoo.SimpleModel import SimpleModel
from PBMF.attention.model_zoo.SimpleModel import Net

def compare_PBMF_and_VT(data, test_size = 0.5, seed = 0, pbmf_epochs = 1000, save_freq = 100,
                       time = 'observed_t', event = 'event', treatment = 'trt', biomarker_ground_truth = 'vard', features = [],
                       ignore_patients_frac = 0.0, minp = 0.5, w1 = 1.0, w2 = 0.0, l1 = 0.0, learning_rate = 0.001,
                       outfile = 'test.csv', n_replica = 10, run_name = 'Test', experiment = 'SIMULATIONS', root = '/scratch/kmvr819/data/PBMF/', layers = [64]):
    

    logger = Logger(
    experiment= experiment,
    run_name=run_name,
    root=root,
    )
    run_id = logger.parent_id
    print(run_id)

    logger = Logger(
    experiment=experiment,
    run_id=run_id,
    root='/scratch/kmvr819/data/PBMF/',
    )

    results_rows = []

    for seed in tqdm.tqdm(range(n_replica)):
        # Create the train/test split
        data_train, data_test = train_test_split(data, test_size=test_size, random_state=seed, stratify=data[treatment])
        data_train = data_train.reset_index(drop=True).copy()
        data_test = data_test.reset_index(drop=True).copy()


        # Run PBMF
        set_seed(0)
        print(run_name)
        run, artifacts = logger.init(run_name = '{} - {}'.format(run_name, seed))

        pbmf = PBMF()
        pbmf.set_parameters(
            time=time, 
            event=event, 
            treatment=treatment, 
            features=features,
            outdir=artifacts,
            ignore_patients_frac=ignore_patients_frac, # During training, ignore this % of patients before computing the loss
            layers=layers,
            epochs=pbmf_epochs,
            minp=minp, # force the model to get this minimum population
            w1=w1, 
            w2=w2,
            embeddings_dim=32,
            seed=0,
            learning_rate=learning_rate,
            shuffle=True,
            shuffle_features=True,
            l1=l1,
        )

        pbmf.set_data(data_train)
        pbmf.fit(save_freq=save_freq)

        X_train, y_train = pbmf.dataloader.transform(data_train)
        X_test, y_test = pbmf.dataloader.transform(data_test)
        bix = biomarker_index(np.array(y_train), pbmf.predict(data_train)[0])

        data_train['risk'] = pbmf.predict(data_train)[0][:, bix]
        data_test['risk'] = pbmf.predict(data_test)[0][:, bix]

        #METRICS
        cm = get_confusion_matrix2(data_test, biomarker_ground_truth = biomarker_ground_truth)
        _, auc_roc_train, _, auc_pr_train = get_AUC(data_train, true_variable = biomarker_ground_truth)
        _, auc_roc, _, auc_pr = get_AUC(data_test, true_variable = biomarker_ground_truth)
        metrics = calculate_metrics(cm)

        a = pd.DataFrame(cm, index = [seed])
        b1 = pd.DataFrame({'AUC_ROC_train': auc_roc_train, 'AUC_PR_train': auc_pr_train}, index = [seed])
        b = pd.DataFrame({'AUC_ROC': auc_roc, 'AUC_PR': auc_pr}, index = [seed])
        c = pd.DataFrame(metrics, index = [seed])
        row = pd.concat([a, b1, b, c], axis = 1)
        
        # Run VT
        model_args = dict(
        num_trees = 100,
        )

        model_fit_args = dict(
            max_features="sqrt", max_depth=5, min_node_size=10, seed=1
        )

        vt = VirtualTwins(
            RandomSurvivalForestModel, 
            features, time=time, event=event, 
            treatment='trt',
            model_args = model_args,
            model_fit_args = model_fit_args
        )

        vt.fit(data_train)

        data_train['risk_vt'] = -vt.predict(data_train)
        data_test['risk_vt'] = -vt.predict(data_test)

        thr = data_train['risk_vt'].median()
        
        cm_vt = get_confusion_matrix2(data_test, risk_score = 'risk_vt', thr = thr, biomarker_ground_truth = biomarker_ground_truth)
        _, auc_roc_train_vt, _, auc_pr_train_vt = get_AUC(data_train, true_variable = biomarker_ground_truth, risk_score ='risk_vt')
        _, auc_roc_vt, _, auc_pr_vt = get_AUC(data_test, true_variable = biomarker_ground_truth, risk_score ='risk_vt')
        metrics_vt = calculate_metrics(cm_vt)
        
        a_vt = pd.DataFrame(cm_vt, index = [seed])
        b1_vt = pd.DataFrame({'AUC_ROC_train': auc_roc_train_vt, 'AUC_PR_train': auc_pr_train_vt}, index = [seed])
        b_vt = pd.DataFrame({'AUC_ROC': auc_roc_vt, 'AUC_PR': auc_pr_vt}, index = [seed])
        c_vt = pd.DataFrame(metrics_vt, index = [seed])
        row_vt = pd.concat([a_vt, b1_vt, b_vt, c_vt], axis = 1)
        
        row_vt.columns = ['TN_vt', 'FP_vt', 'FN_vt', 'TP_vt', 'AUC_ROC_train_vt', 'AUC_PR_train_vt', 'AUC_ROC_vt',
       'AUC_PR_vt', 'Precision_vt', 'Recall_vt', 'F1_vt', 'MCC_vt',
       'Positive Predicted Class Ratio_vt', 'Accuracy_vt']
        
        # Save results
        results_rows.append(pd.concat([row, row_vt], axis = 1))

        #Save metrics to mlflow
        for key, value in zip(row.iloc[0].index, row.iloc[0].values):
            logger.child.log_metric(key=key, value=value, step = seed)

        logger.child.end_run()

    # Save results
    results_rows = pd.concat(results_rows)
    results_rows.to_csv(outfile)
    
    return results_rows

def compare_PBMF_simple_and_VT(data, test_size = 0.5, seed = 0, pbmf_epochs = 1000, save_freq = 100,
                       time = 'observed_t', event = 'event', treatment = 'trt', biomarker_ground_truth = 'vard', features = [],
                       ignore_patients_frac = 0.0, minp = 0.5, w1 = 1.0, w2 = 0.0, l1 = 0.0, learning_rate = 0.001,
                       outfile = 'test.csv', n_replica = 10, run_name = 'Test', experiment = 'SIMULATIONS', root = '/scratch/kmvr819/data/PBMF/', layers = [64]):
    

    logger = Logger(
    experiment= experiment,
    run_name=run_name,
    root=root,
    )
    run_id = logger.parent_id
    print(run_id)

    logger = Logger(
    experiment=experiment,
    run_id=run_id,
    root='/scratch/kmvr819/data/PBMF/',
    )

    results_rows = []

    for seed in tqdm.tqdm(range(n_replica)):
        # Create the train/test split
        data_train, data_test = train_test_split(data, test_size=test_size, random_state=seed, stratify=data[treatment])
        data_train = data_train.reset_index(drop=True).copy()
        data_test = data_test.reset_index(drop=True).copy()


        # Run PBMF
        set_seed(0)
        print(run_name)
        run, artifacts = logger.init(run_name = '{} - {}'.format(run_name, seed))

        pbmf = PBMF(architecture=Net)
        pbmf.set_parameters(
            time=time, 
            event=event, 
            treatment=treatment, 
            features=features,
            outdir=artifacts,
            ignore_patients_frac=ignore_patients_frac, # During training, ignore this % of patients before computing the loss
            layers=layers,
            epochs=pbmf_epochs,
            minp=minp, # force the model to get this minimum population
            w1=w1, 
            w2=w2,
            embeddings_dim=32,
            seed=0,
            learning_rate=learning_rate,
            shuffle=True,
            shuffle_features=True,
            l1=l1,
        )

        pbmf.set_data(data_train)
        pbmf.fit(save_freq=save_freq)

        X_train, y_train = pbmf.dataloader.transform(data_train)
        X_test, y_test = pbmf.dataloader.transform(data_test)
        bix = biomarker_index(np.array(y_train), pbmf.predict(data_train)[0])

        data_train['risk'] = pbmf.predict(data_train)[0][:, bix]
        data_test['risk'] = pbmf.predict(data_test)[0][:, bix]

        #METRICS
        cm = get_confusion_matrix2(data_test, biomarker_ground_truth = biomarker_ground_truth)
        _, auc_roc_train, _, auc_pr_train = get_AUC(data_train, true_variable = biomarker_ground_truth)
        _, auc_roc, _, auc_pr = get_AUC(data_test, true_variable = biomarker_ground_truth)
        metrics = calculate_metrics(cm)

        a = pd.DataFrame(cm, index = [seed])
        b1 = pd.DataFrame({'AUC_ROC_train': auc_roc_train, 'AUC_PR_train': auc_pr_train}, index = [seed])
        b = pd.DataFrame({'AUC_ROC': auc_roc, 'AUC_PR': auc_pr}, index = [seed])
        c = pd.DataFrame(metrics, index = [seed])
        row = pd.concat([a, b1, b, c], axis = 1)
        
        # Run VT
        model_args = dict(
        num_trees = 100,
        )

        model_fit_args = dict(
            max_features="sqrt", max_depth=5, min_node_size=10, seed=1
        )

        vt = VirtualTwins(
            RandomSurvivalForestModel, 
            features, time=time, event=event, 
            treatment='trt',
            model_args = model_args,
            model_fit_args = model_fit_args
        )

        vt.fit(data_train)

        data_train['risk_vt'] = -vt.predict(data_train)
        data_test['risk_vt'] = -vt.predict(data_test)

        thr = data_train['risk_vt'].median()
        
        cm_vt = get_confusion_matrix2(data_test, risk_score = 'risk_vt', thr = thr, biomarker_ground_truth = biomarker_ground_truth)
        _, auc_roc_train_vt, _, auc_pr_train_vt = get_AUC(data_train, true_variable = biomarker_ground_truth, risk_score ='risk_vt')
        _, auc_roc_vt, _, auc_pr_vt = get_AUC(data_test, true_variable = biomarker_ground_truth, risk_score ='risk_vt')
        metrics_vt = calculate_metrics(cm_vt)
        
        a_vt = pd.DataFrame(cm_vt, index = [seed])
        b1_vt = pd.DataFrame({'AUC_ROC_train': auc_roc_train_vt, 'AUC_PR_train': auc_pr_train_vt}, index = [seed])
        b_vt = pd.DataFrame({'AUC_ROC': auc_roc_vt, 'AUC_PR': auc_pr_vt}, index = [seed])
        c_vt = pd.DataFrame(metrics_vt, index = [seed])
        row_vt = pd.concat([a_vt, b1_vt, b_vt, c_vt], axis = 1)
        
        row_vt.columns = ['TN_vt', 'FP_vt', 'FN_vt', 'TP_vt', 'AUC_ROC_train_vt', 'AUC_PR_train_vt', 'AUC_ROC_vt',
       'AUC_PR_vt', 'Precision_vt', 'Recall_vt', 'F1_vt', 'MCC_vt',
       'Positive Predicted Class Ratio_vt', 'Accuracy_vt']
        
        # Save results
        results_rows.append(pd.concat([row, row_vt], axis = 1))

        #Save metrics to mlflow
        for key, value in zip(row.iloc[0].index, row.iloc[0].values):
            logger.child.log_metric(key=key, value=value, step = seed)

        logger.child.end_run()

    # Save results
    results_rows = pd.concat(results_rows)
    results_rows.to_csv(outfile)
    
    return results_rows

