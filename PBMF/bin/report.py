import yaml
from sklearn import tree
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning

from samecode.plot.pyplot import subplots
from samecode.survival.plot import KMPlot
from matplotlib.backends.backend_pdf import PdfPages

import pickle
import os

import warnings

from samecode.survival.plot import kmf_survival_functions, cox_functions
from samecode.survival.plot import median_plot_survival

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def report(kwargs):

    args = yaml.safe_load(
        open(kwargs.config, 'r')
    )
   
    time = args['outcomes']['time']
    event = args['outcomes']['event']
    treatment = args['outcomes']['treatment']
    control = args['outcomes']['control']
    arms = args['outcomes']['arm_column']
    seed = args['pbmf']['seed']
    replicas = args['pbmf']['replicas']
    epoch = args['results']['epoch']

    # Features
    ft=[]
    for fi in args['features'].values():
        ft+=fi

    # plot params
    t1 = treatment
    ct = control

    KMARGS = dict(
        x_legend = 0.5, y_legend = 0.95, legend_font_size=8,
        comparisons=[
            ['{}_B+'.format(t1), '{}_B+'.format(ct), 'B+: '.format()],
            ['{}_B-'.format(t1), '{}_B-'.format(ct), 'B-: '.format()],
            ['{}_B+'.format(t1), '{}_B-'.format(t1), 'IO: '.format()],
            ['{}_B+'.format(ct), '{}_B-'.format(ct), 'SOC: '.format()]
        ],
        x_hr_legend = 0.0, y_hr_legend = 0.2, hr_font_size=8, 
        hr_color='black',
        linewidth=2,
        template_color = '#8A8F91',
        linestyle=['--', '-', '--', '-'],
        show_censor=True
    )

    # print(data[arms].value_counts())
     
    for data_split in range(args['pbmf']['replicas']):

        artifacts = "{}/{}_data_split".format(args['logger']['run_path'], data_split)
        p = PdfPages('{}/report.pdf'.format(artifacts))


        # Full PBMF
        if False: 
            data_train = pd.read_csv("{}/data_train-epoch-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']))
            data_test = pd.read_csv("{}/data_test-epoch-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']))

            f, axs = subplots(cols=2, rows=1, w=10, h=3, return_f=True)
            KMPlot(data_train, time=time, event=event, label=[arms, 'bin_risk']).plot(
                ax=axs[0],
                title='Training Set',
                **KMARGS
            )

            KMPlot(data_test, time=time, event=event, label=[arms, 'bin_risk']).plot(
                ax=axs[1],
                title='Testing Set',
                **KMARGS
            )
            
            sns.despine(offset=10)
            f.savefig(p, format='pdf')

        # Pruned PBMF
        if False:
            data_train = pd.read_csv("{}/data_train-epoch-{}-pruning-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']))
            data_test = pd.read_csv("{}/data_test-epoch-{}-pruning-{}.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']))

            data_train['pruned_bin_risk'] = (data_train['pruning_consensus_risk_min'] > 0.5).replace([False, True], ['B-', 'B+'])
            data_test['pruned_bin_risk'] = (data_test['pruning_consensus_risk_min'] > 0.5).replace([False, True], ['B-', 'B+'])

            f, axs = subplots(cols=2, rows=1, w=10, h=3, return_f=True)
            KMPlot(data_train, time=time, event=event, label=[arms, 'pruned_bin_risk']).plot(
                ax=axs[0],
                title='Training Set',
                **KMARGS
            )

            KMPlot(data_test, time=time, event=event, label=[arms, 'pruned_bin_risk']).plot(
                ax=axs[1],
                title='Testing Set',
                **KMARGS
            )
            
            sns.despine(offset=10)
            f.savefig(p, format='pdf')

        # Distilled PBMF
        if False:
            data_train = pd.read_csv("{}/data_train-epoch-{}-pruning-{}-distiled.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']))
            data_test = pd.read_csv("{}/data_test-epoch-{}-pruning-{}-distiled.csv".format(artifacts, args['results']['epoch'], args['pruning']['percentile']))

            f, axs = subplots(cols=2, rows=1, w=10, h=3, return_f=True)
            KMPlot(data_train, time=time, event=event, label=[arms, 'tree_predicted_biomarker']).plot(
                ax=axs[0],
                title='Training Set',
                **KMARGS
            )

            KMPlot(data_test, time=time, event=event, label=[arms, 'tree_predicted_biomarker']).plot(
                ax=axs[1],
                title='Testing Set',
                **KMARGS
            )
            
            sns.despine(offset=10)
            f.savefig(p, format='pdf')

        # Tree
        if True:
            f, axs = subplots(cols=1, rows=1, w=10, h=4, return_f=True)
            clf = pickle.load(open('{}/tree.pk'.format(artifacts), 'rb'))

            text_tree = tree.plot_tree(clf, 
                    feature_names=ft,  
                    class_names=['B+', 'B-'],
                    filled=True, ax=axs[0]
            )

            f.savefig(p, format='pdf')

            p.close()

    warnings.filterwarnings("error")
    # Get Hazard Ratio Report 
    path = args['logger']['run_path']

    files = [
        ['{}_data_split/data_train-epoch-{}.csv', '{}_data_split/data_test-epoch-{}.csv', 'PBMF', 'consensus_risk'],
        ['{}_data_split/data_train-epoch-{}' + '-pruning-{}.csv'.format(args['pruning']['percentile']), '{}_data_split/data_test-epoch-{}' + '-pruning-{}.csv'.format(args['pruning']['percentile']), 'PBMF+PRUNING', 'pruning_consensus_risk_min'],
        ['{}_data_split/data_train-epoch-{}' + '-pruning-{}-distiled.csv'.format(args['pruning']['percentile']), '{}_data_split/data_test-epoch-{}' + '-pruning-{}-distiled.csv'.format(args['pruning']['percentile']), 'PBMF+PRUNING+DISTIL', 'tree_predicted_biomarker'],
    ]
    
    p_hr = PdfPages('{}/report_hr-epoch-{}-pruning-{}.pdf'.format(path, args['results']['epoch'], args['pruning']['percentile']))

    for tr, ts, name, risk in files:
        hr_data = hr_boxplot(base_path=path, 
                                time=time, 
                                event=event, 
                                replicas=replicas,epoch=epoch,
                                label=[arms,'bin_risk'],
                                t1=treatment,
                                t2 = control,
                                file_path_train = tr,
                                file_path_test = ts,
                                risk=risk
        )

        
        f, axs = subplots(cols=1, rows=1, w=10, h=4, return_f=True)

        sns.boxplot(x='value', y='variable', data=pd.melt(hr_data), ax=axs[0]).set(
                title = '', xlabel='HR', ylabel='')
        axs[0].set_title(name)
        
        f.savefig(p_hr, format='pdf')


    p_hr.close()
        


def get_HR(data, time, event, label, comparison):
    """
    Helper function to return hazard ratio for a given dataset 
    """
    # combine labels to form comparisons 
    # i.e. SOC_B+ and SOC_B- ect 
    data = data.copy()    
    data['__label__'] = data[label].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    tar = comparison[0]
    ref = comparison[1]

    # only get rows with comparison we are interested in 
    x = data[data.__label__.isin([tar, ref])][[time, event, '__label__']].copy().reset_index(drop=True)
    x.__label__.replace(ref, 0, inplace=True)
    x.__label__.replace(tar, 1, inplace=True)
    x.__label__ = x.__label__.astype(float)

    cph = CoxPHFitter().fit(x, duration_col = time, event_col = event) 
    cph = cph.summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].reset_index().to_dict()
    return cph.get('exp(coef)').get(0)

def hr_boxplot(time, event, replicas, 
               label=['ARM', 'bin_risk'], 
               t1 ='SOC+D', 
               t2='SOC+D+T',
               epoch=100,
               base_path = 'clinical_bloodRNA',
               file_path_train = '',
               file_path_test = '',
               risk=''
    ):
    
    hrs_bplus_train = []
    hrs_bminus_train = []
    hrs_bplus_test = []
    hrs_bminus_test = []
    
    for model in range(replicas):
        
        # open results 
        path_train = os.path.join(base_path, file_path_train.format(model, epoch))
        path_test = os.path.join(base_path, file_path_test.format(model, epoch))
        
        data_train = pd.read_csv(path_train, low_memory=False)
        data_test = pd.read_csv(path_test, low_memory=False)

        if risk != 'tree_predicted_biomarker':
            data_train['bin_risk'] = (data_train[risk] > 0.5).replace([False, True], ['B-', 'B+'])
            data_test['bin_risk'] = (data_test[risk] > 0.5).replace([False, True], ['B-', 'B+'])
        else:
            data_train['bin_risk'] = data_train['tree_predicted_biomarker']
            data_test['bin_risk'] = data_test['tree_predicted_biomarker']


        # calculate Hazard Ratios
        try:
            hr_bplus_train = get_HR(data_train, time, event, label, comparison=['{}_B+'.format(t1), '{}_B+'.format(t2)])
            hr_bminus_train = get_HR(data_train, time, event, label, comparison=['{}_B-'.format(t1), '{}_B-'.format(t2)])

            hr_bplus_test = get_HR(data_test, time, event, label, comparison=['{}_B+'.format(t1), '{}_B+'.format(t2)])
            hr_bminus_test = get_HR(data_test, time, event, label, comparison=['{}_B-'.format(t1), '{}_B-'.format(t2)])

            hrs_bplus_train.append(hr_bplus_train)
            hrs_bminus_train.append(hr_bminus_train)
            hrs_bplus_test.append(hr_bplus_test)
            hrs_bminus_test.append(hr_bminus_test)  
        except ConvergenceWarning:

            data_train['__label__'] = data_train[label].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            print(data_train.value_counts('__label__'))
            data_test['__label__'] = data_test[label].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            print(data_test.value_counts('__label__'))
        except ConvergenceError:
            print('CoxPH Convergence Error, HR will not be included in output')
        
    
    hr_data = pd.DataFrame({
        'HR B+ Train': hrs_bplus_train,
        'HR B- Train': hrs_bminus_train,
        'HR B+ Test': hrs_bplus_test,
        'HR B- Test': hrs_bminus_test
    })
    
    return hr_data


def get_results(config_file, runtype='pbmf', query=None):
    
    if runtype == 'pbmf':
        file_str = '{}/{}_data_split/data_{}-epoch-{}.csv'
        bin_risk = 'bin_risk'
    elif runtype == 'pruning': 
        file_str = '{}/{}_data_split/data_{}-epoch-{}-pruning-{}.csv'
        bin_risk = 'pruned_bin_risk'
    elif runtype == 'distil':
        file_str = '{}/{}_data_split/data_{}-epoch-{}-pruning-{}-distiled.csv'
        bin_risk = 'tree_predicted_biomarker'
        
    
    config = yaml.safe_load(open(config_file))
                
    run_path = config['logger']['run_path']
    epoch = config['results']['epoch']
    time = config['outcomes']['time']
    event = config['outcomes']['event']
    treatment = config['outcomes']['arm_column']
    pruning = config['pruning']['percentile']
    treatment_arm = config['outcomes']['treatment']
    control_arm = config['outcomes']['control']

    tests = []
    trains = []
    for replicate in range(config['pbmf']['replicas']):
        try:
            test_data = pd.read_csv(file_str.format(run_path, replicate, 'test', epoch, pruning), low_memory=False)
            test_data['fold'] = replicate

            # display(eval('''test_data[(test_data['PDL1'] <= 1)]'''))

            if query:
                stri = '''{}'''.format(query.replace('df', 'test_data'))
                # print(stri)
                test_data = eval(stri)
            
            tests.append(test_data)

            train_data = pd.read_csv(file_str.format(run_path, replicate, 'train', epoch, pruning), low_memory=False)
            train_data['fold'] = replicate
            if query:
                train_data = eval('''{}'''.format(query.replace('df', 'train_data')))
            trains.append(train_data)
            
            if runtype == 'pruning':
                train_data['pruned_bin_risk'] = (train_data['pruning_consensus_risk_min'] > 0.5).replace([False, True], ['B-', 'B+'])
                test_data['pruned_bin_risk'] = (test_data['pruning_consensus_risk_min'] > 0.5).replace([False, True], ['B-', 'B+'])

        except Exception as inst:
            print('Replicate {} has issues and it is ignored!'.format(replicate))
            print(str(inst))


    trains = pd.concat(trains).reset_index(drop=True)[[time, event, treatment, bin_risk, 'fold']]
    tests = pd.concat(tests).reset_index(drop=True)[[time, event, treatment, bin_risk, 'fold']]

    train_kmfs = kmf_survival_functions(trains, iteration_column='fold', predictor=[bin_risk, treatment], time=time, event=event)
    test_kmfs = kmf_survival_functions(tests, iteration_column='fold', predictor=[bin_risk, treatment], time=time, event=event)

    cox1 = cox_functions(tests[tests[bin_risk] == 'B+'].reset_index(drop=True), iteration_column='fold', predictor=[treatment], control_arm_label=control_arm, time=time, event=event)
    cox1['comparison'] = 'B+'

    cox2 = cox_functions(tests[tests[bin_risk] == 'B-'].reset_index(drop=True), iteration_column='fold', predictor=[treatment], control_arm_label=control_arm, time=time, event=event)
    cox2['comparison'] = 'B-'
    cox_test = pd.concat([cox1, cox2])
    cox_test['experiment'] = "+".join(config_file.replace('.yaml', '').split('-')[3:])
    cox_test_mean = cox_test.groupby('comparison').median().reset_index()
    cox_test_mean['experiment'] = "+".join(config_file.replace('.yaml', '').split('-')[3:])
    cox_test_mean['data_split'] = 'test'
    
    
    cox1 = cox_functions(trains[trains[bin_risk] == 'B+'].reset_index(drop=True), iteration_column='fold', predictor=[treatment], control_arm_label=control_arm, time=time, event=event)
    cox1['comparison'] = 'B+'
    cox2 = cox_functions(trains[trains[bin_risk] == 'B-'].reset_index(drop=True), iteration_column='fold', predictor=[treatment], control_arm_label=control_arm, time=time, event=event)
    cox2['comparison'] = 'B-'
    cox_train = pd.concat([cox1, cox2])
    cox_train['experiment'] = "+".join(config_file.replace('.yaml', '').split('-')[3:])

    cox_train_mean = cox_train.groupby('comparison').median().reset_index()
    cox_train_mean['experiment'] = "+".join(config_file.replace('.yaml', '').split('-')[3:])
    cox_train_mean['data_split'] = 'train'
    
    
    return  pd.concat([cox_train_mean, cox_test_mean]).reset_index(drop=True), pd.concat([cox_train, cox_test]).reset_index(drop=True)