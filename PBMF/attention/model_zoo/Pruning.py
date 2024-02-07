import yaml
import pandas as pd
import numpy as np
from itertools import combinations
#import multiprocess
import multiprocessing

def get_models(outdir, nmodels = 128):
    # This functions gets the variables in a particular model
    # outdir (string): directory where the models has saved
    # nmodels (int): number of models in the ensemble (default = 128)
    #
    # Usage:
    # get_models('runs_41/replica0')
    # Returns a list of variables for each models (list of lists)
    #
    metrics = []
    for i in range(nmodels):
        f = yaml.load(open('{}/{}/parameters.yaml'.format(outdir, str(i))), Loader=yaml.SafeLoader)
        f = f['dataloader_params']['features']
        metrics.append(f)

    return metrics

def convert_models(df, features):
    # Given a dataframe containing the variables of N models, transform the dataframe to a binary one
    # using the features in the list features.
    # df : dataframe with models
    # features: list of features
    #
    # Usage
    #
    # features = ['var{}'.format(i) for i in range(1,21)]
    # models = convert_models(pd.DataFrame(get_models('runs_41/replica0')), features)
    
    rows_converted = []
    for i in range(len(df)):
        row = df.iloc[i].values
        row_converted = [1 if feat in row else 0 for feat in features]
        rows_converted.append(row_converted)
    
    rows_converted = pd.DataFrame(rows_converted, index = range(len(df)))
    rows_converted.columns = features
    return rows_converted

def compute_consensus(df, risks_columns, thr = 0.5):
    # Given the risks of the ensemble, a consensus matrix for the patients is calculated
    # df: dataframe containing the risks (patients x features)
    # risks_columns: list containing the names of the risks columns
    # thr: threshold for binarizing the risk in B- and B+ (default = 0.5)
    #
    # USAGE:
    #
    # res = compute_consensus(data_train, ['risk_seed{}'.format(i) for i in range(128)], thr = 0.5)
    
    data_train_proc = df[risks_columns].copy()
    data_train_proc = (data_train_proc > thr).astype(int)
    
    res = np.eye(df.shape[0])
    els = combinations(range(df.shape[0]), 2)
    
    for i,j in els:
        res[i,j] = (data_train_proc.iloc[i] == data_train_proc.iloc[j]).mean()
        res[j,i] = res[i, j]
    
    return res

def compute_consensus_mini(df, els, thr = 0.5):
    
    res = {}
    
    for el in els:
        i,j = el[0], el[1]
        res[(i,j)] = (df.iloc[i] == df.iloc[j]).mean()
    return res

def compute_consensus_parallel(df, risks_columns, thr = 0.5, n_jobs =1):
    # Given the risks of the ensemble, a consensus matrix for the patients is calculated
    # df: dataframe containing the risks (patients x features)
    # risks_columns: list containing the names of the risks columns
    # thr: threshold for binarizing the risk in B- and B+ (default = 0.5)
    #
    # USAGE:
    #
    # res = compute_consensus(data_train, ['risk_seed{}'.format(i) for i in range(128)], thr = 0.5)
    
    
    if n_jobs == 1:
        return compute_consensus(df = df, risks_columns = risks_columns, thr = thr)
    
    data_train_proc = df[risks_columns].copy()
    data_train_proc = (data_train_proc > thr).astype(int)
    
    els = list(combinations(range(df.shape[0]), 2))
    np.random.shuffle(els)
    els_split = np.array_split(els, n_jobs)
    
    arg_iterable = [(data_train_proc, els, thr) for els in els_split]
#    with multiprocess.Pool(n_jobs) as pool:
    with multiprocessing.Pool(n_jobs) as pool:
        res = pool.starmap(compute_consensus_mini, arg_iterable)
        
    cm = np.eye(df.shape[0])
    for r in res:
        for k in r.keys():
            i, j = k[0],k[1]
            cm[i,j] = r[k]
            cm[j,i] = r[k]
    
    return cm


def compute_correlations(df, risk_columns, consensus_matrix):
    # Computes the correlation between the risks of the models and the consensus matrix
    # df: dataframe containing the risks in the columns
    # risk_columns: list of the names of the columns containing the risks
    # consensus_matrix: matrix computed using compute_consensus function
    #
    # USAGE:
    #
    # res_corr = compute_correlations(data_train, ['risk_seed{}'.format(i) for i in range(128)], res1)
    
    res_corr = []
    df_tmp = df[risk_columns].copy()
    
    for i in range(df.shape[0]):
        df_tmp['res'] = consensus_matrix[:,i]
        res_corr.append(df_tmp.corr()['res'].values)
        
    res_corr = pd.DataFrame(res_corr)
    res_corr = res_corr.iloc[:,:-1]
    
    return res_corr

def compute_correlations_parallel(df, risk_columns, consensus_matrix, n_jobs = 1):
    # Computes the correlation between the risks of the models and the consensus matrix
    # df: dataframe containing the risks in the columns
    # risk_columns: list of the names of the columns containing the risks
    # consensus_matrix: matrix computed using compute_consensus function
    #
    # USAGE:
    #
    # res_corr = compute_correlations(data_train, ['risk_seed{}'.format(i) for i in range(128)], res1)
    
    if n_jobs == 1:
        return compute_correlations(df, risk_columns, consensus_matrix)
    
    res_corr = []
    df_tmp = df[risk_columns].copy()
    
    patient_list = list(df.index.values)
    patient_list_split = np.array_split(patient_list, n_jobs)
    
    arg_iterable = [(df_tmp, consensus_matrix, patients) for patients in patient_list_split]
    with multiprocessing.Pool(n_jobs) as pool:
        res = pool.starmap(compute_correlations_mini, arg_iterable)
    return pd.concat(res)

def compute_correlations_mini(df, consensus_matrix, patients):
    
    res_corr = []
    df_tmp = df.copy()
    #print(patients)
    
    for i in patients:
        df_tmp['res'] = consensus_matrix[:,i]
        res_corr.append(df_tmp.corr()['res'].values)
        
    res_corr = pd.DataFrame(res_corr)
    res_corr = res_corr.iloc[:,:-1]
    res_corr.set_index(patients, inplace = True)
    
    return res_corr

def select_models(res_corr, drop_negatives = True, select_percentile = 90):
    # Uses the correlation obtained from compute_correlations to select the best models in 
    # general agreement between each other
    #
    # res_corr: mean correlation_matrix between the consensus and the risks
    # drop_negatives (boolean): drop the correlation below 0. Pay attention on the positive side of the correlation matrix (default True)
    # select_percentile: (0 to 100) filter the models with correlation higher than the percentile selected. Once all the models with high correlation
    # are selected, select the most frequent models (using the same percentile). In a nutshell, controls how many models will be used at the end. The default is 
    # is 90, which for 128 initial models selects the best 13.
    #
    # Returns a list of models (which correspond to the original ones in the PBMF directories)
    #
    # USAGE:
    #
    # final_models = select_models(res_corr, drop_negatives = True, select_percentile = 90)
    
    values = res_corr.values.flatten()
    if drop_negatives:
        thr = np.percentile(values[values >= 0], select_percentile)
    else:
        thr = np.percentile(values, select_percentile)
    
    res = (res_corr >= thr).sum().sort_values(ascending = False)
    res = res[res >= np.percentile(res.values, select_percentile)]
   
    return res.index.values