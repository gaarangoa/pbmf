import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import norm
from lifelines import KaplanMeierFitter
import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm

def covariate_creator(trt_p=0.5, means=[0, 0], covs=None, binarize=True, seed=42, size=1000, p=[0.5, 0.5]):
    # This function creates the covariate matrix X for the simulation. The input parameters are as follows:
    # trt_p: proportion of main treatment to be sampled from a binomial distribution (default = 0.5)
    # means: value for the mean of the each variable. It has to be a list of n variables (default 1 variable, mean = 0)
    # covs: covariance matrix to be used in the multivariate normal distribution to produce the covariate matrix. Default is None
    # so the funciton produces an uncorrelated, variance = 1 for all the variables. If provided should a nvar x nvar matrix
    # with the diagonal elements being the variance of each variables and the off elements the corresponding covariances. the matrix is symmetric
    # binarize: whether to transform the normal variables or not using the mean of each one (default True)
    # seed: number for reproducibility
    # size: number of patient to simulate
    # p: percentile for binarization
    #
    # returns:
    # X a covariate matrix

    np.random.seed(seed)
    trt = np.random.binomial(1, p=trt_p, size=size)

    x_means = np.array(means)

    if covs is None:
        # Default is completely uncorrelated variance = 1 for all the variables
        x_covs = np.eye(len(means))
    else:
        x_covs = covs

    X = multivariate_normal.rvs(mean=x_means, cov=x_covs, size=size)

    # Create the dataframe to output
    df = {}
    df['trt'] = trt

    for i in range(len(means)):
        df['var{}'.format(i + 1)] = X[:, i]

    df = pd.DataFrame(df)

    if binarize:
        for i in range(len(df.columns) - 1):
            v = df.columns[i + 1]
            df[v] = (df[v] > norm.ppf(1 - p[i], x_means[i], x_covs[i, i])).astype(int)

    return df


def covariate_extend(df):
    nvar = len(df.columns) - 1

    # create all the treatment x variables:
    for i in range(nvar):
        df['trtxvar{}'.format(i + 1)] = df['trt'] * df['var{}'.format(i + 1)]

    # create all the variable_i x variable_j and its correspondent treatment interaction:
    pairs = []
    for i in range(nvar):
        for j in range(i):
            pairs.append((j + 1, i + 1))
    pairs.sort()
    #print(pairs)

    for pair in pairs:
        i, j = pair
        df['var{}xvar{}'.format(i, j)] = df['var{}'.format(i)] * df['var{}'.format(j)]

    for pair in pairs:
        i, j = pair
        df['trtxvar{}xvar{}'.format(i, j)] = df['trt'] * df['var{}'.format(i)] * df['var{}'.format(j)]

    return df

def censolve(b, df, censoring_ratio):

    #auxiliary function to decide the time to consider for censoring in order to attain a proper censoring ratio

    t_bigger_than_b = df['t'] >= b
    t_lower_than_b = df['t'] < b

    return np.abs(censoring_ratio - (sum(t_bigger_than_b) + sum(df[t_lower_than_b]['t'])/b)/len(df))

def simulate_and_censor(X, betas, gam = 1.5, lamb =0.07, censoring_ratio = 0.05, normalize_to_time = 64,
                        recruiting_timeframe = 18, administrative_date =48, seed = 43, verbose = False):

    # This function creates the simulated times and events from the covariates matrix
    # X: covariates matrix
    # betas: vector of betas to create the proper parameters for the HR
    # gam: parameter for the weibull distribution function of the base hazard (default = 1.5)
    # lamb: parameter for the weibull distribution function of the base hazard (default = 0.07)
    # censoring_ratio: ratio of censoring (default = 0.05)
    # normalize_to_time: time to scale the times. Default is 64, which means that the maximum time while be 64
    # or close to 64. If 0, no normalization is done
    # recruiting_timeframe: virtual date (default 18) used to create an anchor for administrative censoring
    # administrative_date: virtual final date (default 48), counted from the recruiting time + observed time,
    # to create administrative censored data.
    # seed: seed for reproducibility (DO NOT USE THE SAME SEED AS IN covariate_creator()!!!!!!)

    # returns a df containinf the X covariates, observed_t (times) and events

    np.random.seed(seed)

    # Simulate times
    if verbose:
        print('Creating times...')
    df = X.copy()
    den = lamb * np.exp(np.matmul(X.values, betas))
    t = np.array([(-np.log(np.random.uniform()) / i) ** (1.0 / gam) for i in den])
    t = t.flatten()
    df['t'] = t

    if verbose:
        print('times generated')

    if censoring_ratio > 0.0:
        if verbose:
            print('Creating censoring...')
        # Get censored times
        maxt = df['t'].max()
        b = np.linspace(0.01, 100 * maxt, 2000)
        b_for_censor = b[np.argmin([censolve(bi, df, censoring_ratio) for bi in b])]

        df['censored_t'] = np.random.uniform(low=0, high=b_for_censor, size=len(df['t']))
        df['event'] = (df['censored_t'] > df['t']).astype('int')
        df['observed_t'] = [obs if ev == 1 else cens for (obs, cens, ev) in zip(df['t'], df['censored_t'], df['event'])]
        if verbose:
            print('Censoring created...')
    else:
        df['event'] = 1
        df['observed_t'] = df['t']

    # Normalize times and do some administrative censoring
    if normalize_to_time > 0:
        df['observed_t'] = normalize_to_time * df['observed_t'] / df['observed_t'].max()

    # creating a recruiting date
    if administrative_date > 0:
        if verbose:
            print('Creating administrative censoring...')
        df['recruiting_date'] = np.random.uniform(low=0, high=recruiting_timeframe, size=len(df['t']))
        df['observed_date'] = df['recruiting_date'] + df['observed_t']
        # censor the patients passed the administrative date
        df.loc[df['observed_date'] > administrative_date, 'event'] = 0

    return df

def binary_to_continous(df, variables_to_transform):
    # Transform all the var = 0 to negative instances of random normal and all the var = 1 to positive ones
    # df: binarized df with time
    # variables_to_transform: list of variables to transform

    for i in variables_to_transform:
        df.loc[df[i] == 0, i] = -np.abs(np.random.normal(loc = 0.0, scale = 1.0, size = len(df[df[i] == 0].loc[:, i])))
        df.loc[df[i] == 1, i] = np.abs(np.random.normal(loc = 0.0, scale = 1.0, size = len(df[df[i] == 1].loc[:, i])))
    return df


def binary_to_continous2(df, variables_to_transform, normalize = False):
    # Using the proportion of 0/1, create an artificial conitnuous uniform distribution
    # df: binarized df with time
    # variables_to_transform: list of variables to transform

    for i in variables_to_transform:
        m = df[i].mean()
        size0 = len(df[df[i] == 0].loc[:, i])
        size1 = len(df[df[i] == 1].loc[:, i])
        df.loc[df[i] == 0, i] = -np.random.uniform(low = 0.0, high = size0/(size0+size1), size = size0)
        df.loc[df[i] == 1, i] = np.random.uniform(low = 0.0, high = size1/(size0+size1), size = size1)

    if normalize:
        for i in variables_to_transform:
            # center and scale the uniform distribution, add a small noise to fix the limits and then transform to a normal N(0,1)
            tmp = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
            tmp = tmp + 1e-6 * np.random.normal(size=len(tmp))
            tmp2 = norm.ppf(tmp)
            df.loc[:, i] = tmp2
    return df

def create_correlated_data_1D(data, distribution='normal', rho=0.7, seed=42):
    np.random.seed(seed)

    size = len(data)

    # Create a random vector
    if distribution == 'normal':
        X = np.random.normal(size=size)
    elif distribution == 'uniform':
        X = np.random.uniform(size=size)
    else:
        X = np.random.binomial(n=1, p=0.5, size=size)

    # Create a copy of the data
    Y = data.copy()

    # Add a constant for the linear model, compute the models between the data and the new random vector
    Y = sm.add_constant(Y)
    model = sm.OLS(X, Y)
    results = model.fit()

    # Get the residuals
    residuals = results.fittedvalues - X

    # Create the new data with teh corresponding correlation
    Xnew = rho * np.std(residuals) * Y[:, 1] + np.sqrt((1 - rho ** 2)) * np.std(Y[:, 1]) * residuals

    return Xnew
