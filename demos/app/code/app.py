import marimo

__generated_with = "0.9.20"
app = marimo.App(
    width="full",
    app_title="PBMF",
    layout_file="layouts/visualization.grid.json",
)


@app.cell
def __():
    import marimo as mo
    import io
    import sys
    import os

    return io, mo, os, sys


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from tqdm.auto import tqdm
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 500)

    from collections import Counter

    import matplotlib 
    matplotlib.rcParams['figure.dpi'] = 300

    import matplotlib.pyplot as plt
    from collections import Counter
    return Counter, matplotlib, np, pd, plt, sns, tqdm


@app.cell
def __():
    from samecode.plot.pyplot import subplots
    from samecode.survival.plot import KMPlot
    from samecode.random import set_seed
    import pickle
    import yaml
    return KMPlot, pickle, set_seed, subplots, yaml


@app.cell
def __():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    from sklearn.metrics import classification_report
    from sklearn import metrics
    return DecisionTreeClassifier, classification_report, metrics, plot_tree


@app.cell
def __(
    DecisionTreeClassifier,
    KMPlot,
    control,
    data_train_tree,
    event,
    features,
    metrics,
    pd,
    test,
    time,
    train,
    treat,
    treatment,
):
    def generate_predictions(depth=10, relative_to=None):
        if relative_to.value == 'Full PBMF':
            classifier = 'bin_risk'
        else:
            classifier = 'pruning_bin_risk'

        results = []
        hrs = []
        for max_depth in range(1, depth): 
            clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
            clf.fit(
                data_train_tree[features].fillna(0),
                data_train_tree[classifier]
            )

            train['tree_bin_risk'] = clf.predict(train[features].fillna(0))
            test['tree_bin_risk'] = clf.predict(test[features].fillna(0))

            y_true = train[classifier].replace(['B-', 'B+'], [0, 1]).values.astype(int)
            y_pred = train['tree_bin_risk'].replace(['B-', 'B+'], [0, 1]).values.astype(int)


            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            train_auc = metrics.auc(fpr, tpr)

            y_true = test[classifier].replace(['B-', 'B+'], [0, 1]).values.astype(int)
            y_pred = test['tree_bin_risk'].replace(['B-', 'B+'], [0, 1]).values.astype(int)


            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
            test_auc = metrics.auc(fpr, tpr)

            results.append({
                'max_depth': str(max_depth),
                'train_auc': train_auc,
                'test_auc': test_auc
            })

            for split, name in [[train, 'Train'], [test, 'Test']]:
                try:
                    km = KMPlot(
                        split, 
                        time=time, 
                        event=event,
                        label=[treatment, 'tree_bin_risk']
                    )
                    k1 = km.extract_hr(to_compare = [[f'{treat}_B+', f'{control}_B+', 'B+']])[0][1][2]
                    k2 = km.extract_hr(to_compare = [[f'{treat}_B-', f'{control}_B-', 'B-']])[0][1][2]
                    k = pd.DataFrame([k1, k2])
                    k['depth'] = str(max_depth)
                    k['dataset'] = name
                    k['biomarker'] = [i.split('_')[1] for i in k.Treatment]
                    k['Treatment'] = [i.split('_')[0] for i in k.Treatment]
                    k['Control'] = [i.split('_')[0] for i in k.Control]
                    hrs.append(k)
                except:
                    pass

        return pd.DataFrame(results), pd.concat(hrs).reset_index(drop=True)
    return (generate_predictions,)


@app.cell
def __(
    KMPlot,
    control,
    event,
    features,
    kmargs,
    sns,
    subplots,
    test,
    time,
    train,
    treat,
    treatment,
):
    def plot_kms(clf):
        train['tree_bin_risk'] = clf.predict(train[features].fillna(0))
        test['tree_bin_risk'] = clf.predict(test[features].fillna(0))

        f, axs = subplots(cols=3, rows=2, w=16, h=6, return_f=True)
        order = [f'{treat}_B+', f'{treat}_B-', f'{control}_B+', f'{control}_B-', ]
        KMPlot(train, time=time, event=event, label=[treatment, 'bin_risk']).plot(
            order,
            ax=axs[0],
            title='Original (train)',
            **kmargs
        );

        KMPlot(test, time=time, event=event, label=[treatment, 'bin_risk']).plot(
            order,
            ax=axs[3],
            title='test',
            **kmargs
        );

        KMPlot(train, time=time, event=event, label=[treatment, 'pruning_bin_risk']).plot(
            order,
            ax=axs[1],
            title='Pruning (train)',
            **kmargs
        );

        KMPlot(test, time=time, event=event, label=[treatment, 'pruning_bin_risk']).plot(
            order,
            ax=axs[4],
            title='test',
            **kmargs
        );

        KMPlot(train, time=time, event=event, label=[treatment, 'tree_bin_risk']).plot(
            order,
            ax=axs[2],
            title = 'Tree (train)',
            **kmargs
        );

        KMPlot(test, time=time, event=event, label=[treatment, 'tree_bin_risk']).plot(
            order,
            ax=axs[5],
            title='test',
            **kmargs
        );


        sns.despine(offset=10)

        return axs, f
    return (plot_kms,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # PBMF tree distilation experiments
        In this notebook we examine the results provided by the PBMF across all datasets. 
        The purpose of this analysis is to provide a set of trees derived of the PBMF that will be used for interpretability.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Load dataset
        Select one of the datasets for visualization
        """
    )
    return


@app.cell
def __(mo):
    cohorts = mo.ui.dropdown(
        options=[
            "brca", "retinopathy",
            "immotion", "javelin",
            "oak", "checkmate", "imvigor",
        ], value="oak", label=""
    )

    cohorts
    return (cohorts,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Build tree relative to?
        Use pseudo labels to train decision tree classifier from full PBMF or pruned PBMF.
        """
    )
    return


@app.cell
def __(mo):
    relative_to = mo.ui.dropdown(
        options=["Full PBMF", "Pruning"], value="Pruning", label="Relative to: "
    )

    relative_to
    return (relative_to,)


@app.cell
def __(relative_to):
    relative_to.value
    return


@app.cell
def __(cohorts):
    cohort = cohorts.value
    return (cohort,)


@app.cell
def __(cohort, pickle):
    train = pickle.load(open('../data/dataset.pk', 'rb'))[(f'{cohort}','train')].fillna(0)
    test = pickle.load(open('../data/dataset.pk', 'rb'))[(f'{cohort}','test')].fillna(0)
    return test, train


@app.cell
def __(cohort, yaml):
    config = yaml.safe_load(open(f'../data/{cohort}.yaml', 'r'))
    return (config,)


@app.cell
def __(config):
    time = config['time']
    event = config['event']
    treatment = config['treatment']
    control = config['control']
    treat = config['treat']
    features = config['features']

    colors = ['#e66101', '#5e3c99',  'orange',  'blue']
    comparisons=[
            ['{}_B+'.format(treat), '{}_B+'.format(control), 'B+:  '.format()],
            ['{}_B-'.format(treat), '{}_B-'.format(control), 'B-:  '.format()],
        ]

    kmargs = dict(
            x_legend = 0.3, 
            y_legend = 0.95, 
            legend_font_size=8,
            hr_font_size=8,
            # hr_color = 'black',
            template_color='gray',
            show_censor=False,
            linewidth=2,
            comparisons=comparisons,
            colors=colors,
            linestyle=['-', '-', '--', '--'],
            y_hr_legend=0.1
    )
    return (
        colors,
        comparisons,
        control,
        event,
        features,
        kmargs,
        time,
        treat,
        treatment,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Optimize tree depth to recover biomarker
        We train a tree classifier to predict ```B+/B-``` classes. We train multiple tree classifiers using a range of depth from 1 to 10. This parameters indicates a more complex tree and biomarkers. 

        To define the classes we use a ```0.5``` cutoff. Everything above ```0.5``` corresponds to ```B+``` while everything below corresponds to ```B-```.
        """
    )
    return


@app.cell
def __(relative_to, train):
    thr = 0.

    data_train_tree = train[
        ~((train['consensus_risk' if relative_to.value == 'Full PBMF' else 'pruning_risk'] > 0.5-thr) & 
        (train['consensus_risk' if relative_to.value == 'Full PBMF' else 'pruning_risk'] < 0.5+thr))
    ].copy().reset_index(drop=True)
    return data_train_tree, thr


@app.cell
def __(mo):
    mo.md(r"""## Performance at different tree depths""")
    return


@app.cell
def __(generate_predictions, relative_to, sns, subplots):
    results, hrs = generate_predictions(depth=20, relative_to=relative_to)

    def plot_tree_depth():
        f, axs = subplots(cols=1, rows=1, w=6, h=3, return_f=True)
        sns.lineplot(data=results, x='max_depth', y='train_auc', label='Training', ax=axs[0])
        sns.lineplot(data=results, x='max_depth', y='test_auc', label='Testing', ax=axs[0])
        axs[0].set_ylabel('AUC ROC')
        axs[0].set_xlabel('Tree max depth')
        sns.despine();

        return axs

    plot_tree_depth()
    return hrs, plot_tree_depth, results


@app.cell
def __(hrs, sns, subplots):
    def prog2pred_plot():
        axs = subplots(cols=2, rows=1, w=10, h=3)
        sns.lineplot(data=hrs.query('dataset == "Train"'), x='depth', y='HR', hue='biomarker', ax=axs[0], palette=['red', 'blue'])
        sns.lineplot(data=hrs.query('dataset == "Test"'), x='depth', y='HR', hue='biomarker',  ax=axs[1], palette=['red', 'blue'])
        axs[0].set_ylim([0, 4])
        axs[1].set_ylim([0, 4])
        axs[0].set_title('Training set')
        axs[1].set_title('Testing set')

        sns.despine()

        return axs

    prog2pred_plot()
    return (prog2pred_plot,)


@app.cell
def __(mo):
    mo.md(r"""**Figure 1:** Tree classification performance measured using the ROC AUC curve across multiple depths. The task of the tree is to predict if a sample is B+ or B- depending on their input features. We visualize training and testing datsets. Additionally, we extracted the HR scores for each train and test sets through the different depths. This visualization can hihglight the transition from prognostic to predictive.""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## From prognostic to predictive
        Evaluate the impact of tree depth to recapitulate the predictive behavior.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Selection of best parameters
        After defining the ```max depth``` parameter we perform a survival analysis to evaluate its reproducibility. Note that the model can overfit to training and not to testing. Therefore a max depth need to be calculated trying to avoid overfitting.
        """
    )
    return


@app.cell
def __(mo):
    number = mo.ui.number(start=1, stop=20, label="Tree depth", value=3)
    # mo.hstack([number, mo.md(f"Has value: {number.value}")])
    number
    return (number,)


@app.cell
def __(
    DecisionTreeClassifier,
    data_train_tree,
    features,
    number,
    relative_to,
):
    clf = DecisionTreeClassifier(random_state=0, max_depth=number.value)
    clf.fit(
        data_train_tree[features].fillna(0),
        data_train_tree['bin_risk' if relative_to.value == 'Full PBMF' else 'pruning_bin_risk']
    );
    return (clf,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Decision tree visualization
        Useful to inspect the relationship between the different biomarkers.
        """
    )
    return


@app.cell
def __():
    # sns.histplot(x=train['t_cells_cd4_memory_resting'], label='train')
    # sns.histplot(x=test['t_cells_cd4_memory_resting'], label='test')
    return


@app.cell
def __(mo):
    fontsize_tree = mo.ui.number(start=1, stop=28, label="Fontsize", value=10)
    # mo.hstack([number, mo.md(f"Has value: {number.value}")])
    fontsize_tree
    return (fontsize_tree,)


@app.cell
def __(clf, features, fontsize_tree, io, mo, subplots):
    from samecode.plot.tree import plot_tree2

    f1, axs1 = subplots(cols=1, rows=1, w=18, h=8, return_f=True)
    text_tree = plot_tree2(
        clf, 
        feature_names=features, 
        class_names=['B+', 'B-'], 
        class_colors = ['orange', 'darkblue'],
        class_label_colors = ['white', 'white'],
        ax=axs1[0],
        arrow_y_offset=0.0,
        rounded=True,
        fontsize=fontsize_tree.value
    )

    svg_buffer1 = io.StringIO()
    f1.savefig(svg_buffer1, format='svg')
    svg_buffer1.seek(0)
    svg_data1 = svg_buffer1.getvalue()

    # mo.pdf(src=svg_buffer1)

    mo.Html(svg_data1)
    axs1
    return axs1, f1, plot_tree2, svg_buffer1, svg_data1, text_tree


@app.cell
def __():
    return


@app.cell
def __(cohorts, mo, svg_data1):
    download_image = mo.download(
        data=svg_data1,
        filename=f'{cohorts.value}_tree.svg'
    )
    download_image
    return (download_image,)


@app.cell
def __(mo):
    mo.md(r"""**Figure 2:** decision tree classifier. Each node corresponds to a feature, each child node to the left represents the **Yes** value of the condition. Larger trees can be very complex. For interpretability, we could limit to a lower depth.""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Survival Analysis
        Visualize patient stratification and evaluate the biomarker derived from the decision tree
        """
    )
    return


@app.cell
def __(clf, io, mo, plot_kms):
    kmplots, fkm = plot_kms(clf)

    svg_buffer0 = io.StringIO()
    fkm.savefig(svg_buffer0, format='svg')
    svg_buffer0.seek(0)
    svg_data0 = svg_buffer0.getvalue()

    # mo.pdf(src=svg_buffer1)

    mo.Html(svg_data0)
    kmplots
    return fkm, kmplots, svg_buffer0, svg_data0


@app.cell
def __(cohorts, mo, svg_data0):
    download_fplot0 = mo.download(
        data=svg_data0,
        filename=f'{cohorts.value}_kms-train-test-tree.svg'
    )
    download_fplot0
    return (download_fplot0,)


@app.cell
def __(mo):
    mo.md(r"""**Figure 3:** Kapplan-Meier plots for each of the datasets and the different stages of the PBMF (complete model, pruning and tree).""")
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Specific biomarkers""")
    return


@app.cell
def __():
    def query_pts(data, queries):
        group = data
        for query in queries:
            group = group.query(f"{query}")

        return group.index
    return (query_pts,)


@app.cell
def __(
    KMPlot,
    control,
    event,
    kmargs,
    pd,
    query_pts,
    test,
    time,
    train,
    treat,
    treatment,
):
    def km2populations(queries, queries2, axs=None):

        for ix, [split, name] in enumerate([[train, 'Train'], [test, 'Test']]):
            g1 = split.iloc[split.fillna(0).index.isin(query_pts(split, queries))].reset_index(drop=True)
            g1['bmk'] = 'B-'
            g2 = split.iloc[split.fillna(0).index.isin(query_pts(split, queries2))].reset_index(drop=True)
            g2['bmk'] = 'B+'
            g = pd.concat([g1, g2]).reset_index(drop=True)


            order = [f'{treat}_B+', f'{treat}_B-', f'{control}_B+', f'{control}_B-', ]
            KMPlot(g, time=time, event=event, label=[treatment, 'bmk']).plot(
                order,
                ax=axs[ix],
                title=name,
                **kmargs
            );

        return axs
    return (km2populations,)


@app.cell
def __():
    # q1 = "ECOG > 0.5;LivMets > 0.5;`Plasma IL8 (C3D1) (pg/mL)` <= 22.5"
    # q2 = "ECOG < 0.5"

    # km2populations(q1.split(';'), q2.split(';'))
    return


@app.cell
def __(mo):
    mo.md(r"""## Explore a specific biomarker""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Input two decision rules separated by ; for example: 

        * "ECOG > 0.5;LivMets > 0.5 & `Plasma IL8 (C3D1) (pg/mL)` <= 22.5"
        * "ECOG < 0.5"

        These are two populations and they will be compared in the training and testing data.
        For biomarkers containing spaces use `biomarker`
        """
    )
    return


@app.cell
def __():
    # train['del6'] = (train['Deletion_6p22.2'] == 1) | (train['Deletion_6p21.32'] == 1) | (train['Deletion_6q25.2'] == 1) 
    # test['del6'] = (test['Deletion_6p22.2'] == 1) | (test['Deletion_6p21.32'] == 1) | (test['Deletion_6q25.2'] == 1)
    return


@app.cell
def __(mo):
    # MSAF < 0.11;blSLD<140;mutations_MLL2 != 1
    # MSAF > 0.11;mutations_ATM != 1;mutations_PDGFRA != 1

    q1 = mo.ui.text_area(value=" ", label="B- population", placeholder="MSAF < 0.11;blSLD<140;mutations_MLL2 != 1")
    q2 = mo.ui.text_area(value=" ", label="B+ population", placeholder="MSAF > 0.11;mutations_ATM != 1;mutations_PDGFRA != 1")

    q1
    return q1, q2


@app.cell
def __(q2):
    q2
    return


@app.cell
def __():
    return


@app.cell
def __(
    KMPlot,
    control,
    event,
    io,
    kmargs,
    mo,
    pd,
    q1,
    q2,
    sns,
    subplots,
    test,
    time,
    train,
    treat,
    treatment,
):
    f4, ax4 = subplots(cols=1, rows=2, w=5, h=6.5, return_f=True)

                       
    if q1.value != ' ' and q2.value != ' ':
        for ix, [split, name] in enumerate([[train, 'train'], [test, 'test']]):
            split1 = split.query(q1.value).reset_index(drop=True)
            split1['bmk'] = 'B-'
            split2 = split.query(q2.value).reset_index(drop=True)
            split2['bmk'] = 'B+'

            pop = pd.concat([split1, split2]).reset_index(drop=True)
            pop['split'] = name

            order = [f'{treat}_B+', f'{treat}_B-', f'{control}_B+', f'{control}_B-', ]
            KMPlot(pop, time=time, event=event, label=[treatment, 'bmk']).plot(
                    order,
                    ax=ax4[ix],
                    title=name,
                    **kmargs
                );

    sns.despine(offset=10)

    svg_buffer4 = io.StringIO()
    f4.savefig(svg_buffer4, format='svg')
    svg_buffer4.seek(0)
    svg_data4 = svg_buffer4.getvalue()

    mo.Html(svg_data4)
    ax4


    return (
        ax4,
        f4,
        ix,
        name,
        order,
        pop,
        split,
        split1,
        split2,
        svg_buffer4,
        svg_data4,
    )


@app.cell
def __(cohorts, mo, svg_data4):
    download_fplot4 = mo.download(
        data=svg_data4,
        filename=f'{cohorts.value}_km.svg'
    )
    download_fplot4
    return (download_fplot4,)


@app.cell
def __(
    DecisionTreeClassifier,
    KMPlot,
    event,
    features,
    kmargs,
    pd,
    sns,
    subplots,
    test,
    time,
    train,
    treatment,
):
    data = pd.concat([train, test])

    clf2 = DecisionTreeClassifier(random_state=0, max_depth=10)
    clf2.fit(
        data[features].fillna(0),
        data['pruning_bin_risk']
    );

    data['tree_bin_risk'] = clf2.predict(data[features])

    axs2 = subplots(cols=1, rows=1, w=5, h=3, return_f=False)
    KMPlot(data, time=time, event=event, label=[treatment, 'tree_bin_risk']).plot(
        ax=axs2[0],
                **kmargs
            );
    sns.despine(offset=10)
    axs2
    return axs2, clf2, data


@app.cell
def __(clf2, features, fontsize_tree, plot_tree2, subplots):
    f12, axs12 = subplots(cols=1, rows=1, w=18, h=8, return_f=True)
    text_tree2 = plot_tree2(
        clf2, 
        feature_names=features, 
        class_names=['B+', 'B-'], 
        class_colors = ['orange', 'darkblue'],
        class_label_colors = ['white', 'white'],
        ax=axs12[0],
        arrow_y_offset=0.0,
        rounded=True,
        fontsize=fontsize_tree.value
    )

    text_tree2
    return axs12, f12, text_tree2


@app.cell
def __():
    # data['bmk'] = ((data['MSAF'] <= 0.1)).replace([False, True], ['B+', 'B-'])

    # f3, axs3 = subplots(cols=1, rows=1, w=5, h=3, return_f=True)
    # KMPlot(data, time=time, event=event, label=[treatment, 'bmk']).plot(
    #     ax=axs3[0],
    #             **kmargs
    #         );
    # sns.despine(offset=10)
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""Developed and maintained by ADS""")
    return


@app.cell
def __(train):
    train.head()
    return


@app.cell
def __(train):
    train.query('(cfDNA_Input_ng < 25 & MSAF <=0.11) | (btmb < 20 & BAGE > 50) ')
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
