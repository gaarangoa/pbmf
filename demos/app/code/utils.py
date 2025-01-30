def generate_predictions(depth=10):
    results = []
    for max_depth in range(1, depth): 
        clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth, min_samples_split=2, min_samples_leaf=3)
        clf.fit(
            data_train_tree[features].fillna(0),
            data_train_tree.bin_risk
        )

        train['tree_bin_risk'] = clf.predict(train[features].fillna(0))
        test['tree_bin_risk'] = clf.predict(test[features].fillna(0))

        import numpy as np
        from sklearn import metrics

        fpr, tpr, thresholds = metrics.roc_curve(train['bin_risk'].replace(['B-', 'B+'], [0, 1]), train['tree_bin_risk'].replace(['B-', 'B+'], [0, 1]), pos_label=1)
        train_auc = metrics.auc(fpr, tpr)

        fpr, tpr, thresholds = metrics.roc_curve(test['bin_risk'].replace(['B-', 'B+'], [0, 1]), test['tree_bin_risk'].replace(['B-', 'B+'], [0, 1]), pos_label=1)
        test_auc = metrics.auc(fpr, tpr)

        results.append({
            'max_depth': str(max_depth),
            'train_auc': train_auc,
            'test_auc': test_auc
        })
        
    return pd.DataFrame(results)