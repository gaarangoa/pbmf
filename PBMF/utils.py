from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from PBMF.attention.loss import biomarker_index
import numpy as np
import pandas as pd

def get_confusion_matrix(model, X_in, y_in, data_in, iterator, thr = 0.5 , biomarker_ground_truth = 'var1', biomarker = 'bin_risk', B = ['B-','B+']):
    
    data_tmp = data_in.copy()

    bix = biomarker_index(np.array(y_in), model.predict(iterator)[0])
    data_tmp['risk'] = model.predict(iterator)[0][:, bix]
    data_tmp['bin_risk'] = (data_tmp['risk'] > thr).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(data_tmp[biomarker_ground_truth], data_tmp['bin_risk']).ravel()
    
    return {'TN':tn, 'FP':fp, 'FN':fn,'TP':tp}

def get_confusion_matrix2(data_in, risk_score = 'risk', thr = 0.5 , biomarker_ground_truth = 'vard'):
    
    data_tmp = data_in.copy()
    data_tmp['bin_risk'] = (data_tmp[risk_score] > thr).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(data_tmp[biomarker_ground_truth], data_tmp['bin_risk']).ravel()
    
    return {'TN':tn, 'FP':fp, 'FN':fn,'TP':tp}


def compute_AUCROC(data_in, true_variable = 'var1', risk_score ='risk', B = ['B-','B+']):
    #compute the fpr and the tpr
    fpr, tpr, thresholds = roc_curve(data_in[true_variable], data_in[risk_score])
    res = pd.DataFrame([fpr, tpr, thresholds]).transpose()
    res.columns = ['fpr','tpr', 'thr']

    #Compute the AUC_ROC
    auc_roc = auc(fpr, tpr)
    
    return res, auc_roc

def compute_AUCPR(data_in, true_variable = 'var1', risk_score ='risk', B = ['B-','B+']):

    precision, recall, thresholds = precision_recall_curve(data_in[true_variable], data_in[risk_score])
    res = pd.DataFrame([precision, recall, thresholds]).transpose()
    res.columns = ['precision','recall', 'thr']
    
    #Compute the AUC_PR
    auc_pr = auc(recall, precision)
    return res, auc_pr


def get_AUC(data_in, true_variable = 'var1', risk_score ='risk'):
    
    res_aucroc_test, auc_roc_test = compute_AUCROC(data_in, true_variable, risk_score)    
    res_aucpr_test, auc_pr_test = compute_AUCPR(data_in, true_variable, risk_score)
        
    res = [res_aucroc_test, auc_roc_test, res_aucpr_test, auc_pr_test]
    return res

def calculate_Precision(df):
    num = df['TP']
    den = df['TP']+df['FP']
    return num/den

def calculate_Recall(df):
    num = df['TP']
    den = df['TP']+df['FN']
    return num/den

def calculate_F1(df):
    num = 2*df['TP']
    den = 2*df['TP']+df['FP']+df['FN']
    return num/den

def calculate_MCC(df):
    num = (df['TP'] * df['TN'] - df['FP']*df['FN'])
    den = np.sqrt((df['TP']+df['FP'])*(df['TP']+df['FN'])*(df['TN']+df['FP'])*(df['TN']+df['FN']))
    return num/den

def calculate_P(df):
    num = df['TP'] + df['FP']
    den = df['TN'] + df['FP'] +  df['TP'] + df['FN']
    return num/den

def calculate_ACC(df):
    num = df['TP'] + df['TN']
    den = df['TN'] + df['FP'] +  df['TP'] + df['FN']
    return num/den

def calculate_metrics(df):
    res = {
        'Precision':calculate_Precision(df),
        'Recall':calculate_Recall(df),
        'F1':calculate_F1(df),
        'MCC':calculate_MCC(df),
        #'Positive Predicted Class Ratio':calculate_P(df),
        'Accuracy':calculate_ACC(df)
    }
    return res