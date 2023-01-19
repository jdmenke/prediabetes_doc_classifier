import os
import numpy as np
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer

from sklearn_evaluation import plot

import matplotlib.pyplot as plt
import pickle

import random
random.seed(42)
np.random.seed(42)
random_state = 42

#pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)

# Root directory
dirname = os.path.dirname(__file__).rsplit('/', 2)[0]

def normalize_counts(datum):
    adj_datum = np.log(1+datum)
    return adj_datum

def generate_datasets(gran='sectional', target='gold_igt', normal=True, debug=False):
    # Load and process training/testing labels
    lenient_data = pd.read_csv(os.path.join(dirname, 'data/labels_lenient.csv'))

    lenient_data.rename(columns={"Study": "paper"}, inplace=True)
    lenient_data.set_index('paper', inplace=True)
    lenient_data.drop(index=['Samaras 2015', 'Watanabe 2010', 'Jin 2008'], inplace=True)

    if 'sectional' in gran:
        # New data after error analysis -> criteria test patterns (e.g., 2hPG) and multi-level regex patterns
        cai_data = pd.read_csv(os.path.join(dirname, 'data/cai/training_testing_data_sectional.csv'))
        gujral_data = pd.read_csv(os.path.join(dirname, 'data/gujral/training_testing_data_sectional.csv'))
        all_data_sectional = pd.concat([cai_data, gujral_data], ignore_index=True)
        all_data_sectional.set_index('paper', inplace=True)
        processed_lenient_data_sectional = lenient_data.merge(all_data_sectional, how='left', on='paper')
        intermed = processed_lenient_data_sectional.groupby(level=0)
        processed_lenient_data = intermed.last()
        cols2drop = ['a_extracted_sents', 'i_extracted_sents', 'm_extracted_sents', 'r_extracted_sents', 'd_extracted_sents', 'c_extracted_sents', 't_extracted_sents', 'whole_paper']

        if gran == 'sectional_method':
            col_list = processed_lenient_data.columns.to_list()
            for col_i in col_list:
                if not ((col_i.startswith("m_")) or (col_i == 'year')):
                    cols2drop.append(col_i)
        
        if gran == 'sectional_abstract':
            col_list = processed_lenient_data.columns.to_list()
            for col_i in col_list:
                if not ((col_i.startswith("a_")) or (col_i == 'year')):
                    cols2drop.append(col_i)

    else:
        cai_data = pd.read_csv(os.path.join(dirname, 'data/cai/training_testing_data_wp.csv'))
        gujral_data = pd.read_csv(os.path.join(dirname, 'data/gujral/training_testing_data_wp.csv'))
        all_data_wp = pd.concat([cai_data, gujral_data], ignore_index=True)
        all_data_wp.set_index('paper', inplace=True)
        processed_lenient_data_wp = lenient_data.merge(all_data_wp, how='left', on='paper')
        intermed = processed_lenient_data_wp.groupby(level=0)
        processed_lenient_data_wp = intermed.last()
        processed_lenient_data = processed_lenient_data_wp
        cols2drop = []

    if target in 'gold_igt':
        cols4label = ['gold_igt']
    if target in 'gold_ifg_ada':
        cols4label = ['gold_ifg_ada']
    if target in 'gold_ifg_who':
        cols4label = ['gold_ifg_who']
    if target in 'gold_hba1c_ada':
        cols4label = ['gold_hba1c_ada']
    if target in 'gold_hba1c_iec':
        cols4label = ['gold_hba1c_iec']
    
    labels2drop = ['gold_igt', 'gold_ifg_ada', 'gold_ifg_who', 'gold_hba1c_ada', 'gold_hba1c_iec']
    cols2drop.extend(labels2drop)

    # remove paper and year data
    data = processed_lenient_data.drop(columns=cols2drop).reset_index().drop(columns=['paper', 'year'])
    labels = processed_lenient_data[cols4label].reset_index().drop(columns='paper')

    if debug:
        debug_data = processed_lenient_data.drop(columns=cols2drop).reset_index()
        debug_X_train, debug_X_test, debug_y_train, debug_y_test = train_test_split(debug_data, labels, test_size=0.33, random_state=random_state, stratify=labels)
        test_papers = debug_X_test[['paper']]

    if gran == 'restricted_wp':
        data = data.groupby([s.split('_', 1)[-1] for s in data.columns], axis=1).sum().reset_index().drop(columns='index')

    if normal:
        data = data.applymap(normalize_counts)

    data_np = data.to_numpy()
    labels_np = labels.to_numpy()
    labels_np = np.ravel(labels_np)

    if debug:
        return data, data_np, labels_np, test_papers
    
    else:
        return data, data_np, labels_np

def nested_cross_validation(data_np, labels_np, gran, target, trial_no=5):
    performance_log = []
    best_estimators = []
    precision_log = []
    recall_log = []
    f1_log = []
    max_iter = 10000

    # PCA dimensions chosen based on some initial tests
    if gran == 'sectional':
        dim_reduc = ['passthrough', SparsePCA(6), SparsePCA(9)]
    else:
        dim_reduc = ['passthrough', PCA(6)]

    for j in range(trial_no):
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=j)
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=j)
        components = [('scaling', 'passthrough'), ('dimension_reduction', 'passthrough'), ('clf', SVC(random_state=random_state))]

        params = [
            dict(
                scaling=['passthrough', MinMaxScaler()],
                dimension_reduction=dim_reduc,
                clf=[SVC(kernel="linear", random_state=random_state, gamma='auto', shrinking=True)],
                clf__C=[0.5, 1.0, 2.0, 10, 100],
                ),

            dict(
                scaling=['passthrough', MinMaxScaler()],
                dimension_reduction=dim_reduc,
                clf=[RidgeClassifier(positive=True)],
                clf__alpha=[0.1, 1.0, 10],
                clf__tol = [0.01, 0.001, 0.0001],
                ),

            dict(
                scaling=['passthrough', MinMaxScaler()],
                dimension_reduction=dim_reduc,
                clf=[LogisticRegression(random_state=random_state, max_iter=max_iter, solver='liblinear', penalty='l1')],
                clf__C=[0.1, 1.0, 10, 100],
                clf__penalty = ['l1'],
                clf__tol = [0.001, 0.0001, 0.00001],
                ),

            dict(
                scaling=['passthrough'],
                dimension_reduction=['passthrough'],
                clf=[RandomForestClassifier(random_state=random_state)],
                clf__n_estimators=[10, 50, 100, 150],
                clf__criterion = ['gini', 'entropy'],
                clf__max_features = ['sqrt', None],
                clf__bootstrap = [True, False],
                clf__max_depth = [None, 2, 4, 6],
                )

            ]

        best_f1 = 0
        f1_scorer = make_scorer(f1_score)
        for param in params:
            pipe = Pipeline(components)
            gscv = GridSearchCV(pipe, param_grid=param, cv=inner_cv, scoring=f1_scorer, n_jobs=-1, verbose=3)
            ## Used for pairing down gridsearch candidates
            # gscv.fit(data_np, labels_np)
            # plot.grid_search(gscv.cv_results_, change='dimension_reduction', kind='bar', sort=False)
            # plt.title(f"{target} and {gran}")
            # plt.show()
            scores = cross_validate(gscv, X=data_np, y=labels_np, cv=outer_cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
            estimator_name, precision, recall, f1 = (scores['estimator'], scores['test_precision'], scores['test_recall'], scores['test_f1'])
            f1_avg = np.mean(np.array(f1))
            if f1_avg > best_f1:
                best_estimator = estimator_name
                precision_bif = precision # best in fold
                recall_bif = recall
                f1_bif = f1
                best_f1 = np.mean(np.array(f1_bif))
        
        best_estimators.extend(best_estimator)
        precision_log.extend(list(precision_bif))
        recall_log.extend(list(recall_bif))
        f1_log.extend(list(f1_bif))

    performance_log.append([np.mean(np.array(precision_log)), np.std(np.array(precision_log)), np.mean(np.array(recall_log)), np.std(np.array(recall_log)), np.mean(np.array(f1_log)), np.std(np.array(f1_log))])

    return performance_log


if __name__ == "__main__":
    estimate_pipeline_performance = True
    debug_best = False
    targets = ['gold_igt', 'gold_ifg_ada', 'gold_ifg_who', 'gold_hba1c_ada', 'gold_hba1c_iec']
    granularity = ['sectional', 'wp', 'sectional_method', 'sectional_abstract']
    performance_log = []
    for target in targets:
        if not debug_best:
            for gran in granularity:
                data, data_np, labels_np = generate_datasets(gran=gran, target=target)
                log = nested_cross_validation(data_np, labels_np, gran, target)
                performance_log.append((target, gran, log))

    # Performance log contains the parameters and score (f1) of the best performing model across stratified 5 fold dataset
    if estimate_pipeline_performance:
        print(performance_log)