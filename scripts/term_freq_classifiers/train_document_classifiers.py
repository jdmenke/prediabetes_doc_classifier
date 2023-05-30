import os
import numpy as np
import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import pickle
from joblib import dump, load

import random
random.seed(42)
np.random.seed(42)
random_state = 42

# Root directory
dirname = os.path.dirname(__file__).rsplit('/', 2)[0]

def normalize_counts(datum):
    adj_datum = np.log(1+datum)
    return adj_datum

def generate_datasets(gran='sectional', target='gold_igt', normal=True):
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
    data = processed_lenient_data.drop(columns=cols2drop).reset_index()
    data_dict = data['paper'].to_dict()
    data = data.drop(columns=['paper', 'year'])
    #data = processed_lenient_data.drop(columns=cols2drop).reset_index().drop(columns=['paper', 'year'])
    labels = processed_lenient_data[cols4label].reset_index().drop(columns='paper')

    if gran == 'restricted_wp':
        data = data.groupby([s.split('_', 1)[-1] for s in data.columns], axis=1).sum().reset_index().drop(columns='index')

    if normal:
        data = data.applymap(normalize_counts)

    return data_dict, data, labels


def custom_train_test_split(id_dict, data, labels):
    test_set = {'Ares 2019', 'Barr 2007', 'Barzilay 1999', 'Bjarnason 2019', 'Brunner 2010', 'Chien 2008', 'Deedwania 2013', 'Deng 2017', 'Donahue 2011', 'Evans 2015', 'Fang 2019', 'Fox 2005', 'Hajebrahimi 2017', 'Hermanides 2019', 'Hiltunen 2005', 'Hu 2003', 'Hubbard 2019', 'Hunt 2004', 'Janszky 2009', 'Jiang 2020', 'Khang 2010', 'Kim 2008', 'Kim 2014', 'Kim 2017', 'Kowall 2011', 'Laukkanen 2013', 'Lazo-Porras 2020', 'Lu 2019', 'Ma 2012', 'Madssen 2012', 'Mazza 2001', 'McNeill 2005', 'Mirbolouk 2016', 'Mongraw-Chaffin 2017', 'Nakanishi 2004', 'Onat 2005', 'Paprott 2015', 'Parizadeh 2019', 'Rhee 2020', 'Robich 2019', 'Rodriguez 2002', 'Salazar 2016', 'Sui 2011', 'Sung 2009'}
    test_set_ids = []
    for k, v in id_dict.items():
        if v in test_set:
            test_set_ids.append(k)
    train_data = data.loc[~data.index.isin(test_set_ids)].fillna(0).to_numpy()
    test_data = data.iloc[test_set_ids].fillna(0).to_numpy()
    train_labels = labels.loc[~labels.index.isin(test_set_ids)].fillna(0).to_numpy()
    train_labels = np.ravel(train_labels)
    test_labels = labels.iloc[test_set_ids].fillna(0).to_numpy()
    test_labels = np.ravel(test_labels)

    return train_data, test_data, train_labels, test_labels


def cross_validation(data_np, labels_np, gran):
    max_iter = 10000

    # PCA dimensions chosen based on some initial tests
    if gran == 'sectional':
        dim_reduc = ['passthrough', SparsePCA(6), SparsePCA(9)]
    else:
        dim_reduc = ['passthrough', PCA(6)]

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
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
        gscv = GridSearchCV(pipe, param_grid=param, cv=inner_cv, scoring=f1_scorer, n_jobs=-1, verbose=3, refit=True)
        gscv.fit(data_np, labels_np)
        if gscv.best_score_ > best_f1:
            best_estimator = gscv.best_estimator_
            best_f1 = gscv.best_score_

    return best_estimator, best_f1


if __name__ == "__main__":
    estimate_pipeline_performance = True
    targets = ['gold_igt', 'gold_ifg_ada', 'gold_ifg_who', 'gold_hba1c_ada', 'gold_hba1c_iec']
    granularity = ['sectional', 'wp', 'sectional_method', 'sectional_abstract']
    performance_log = []
    for target in targets:
        for gran in granularity:
            data_dict, data, labels = generate_datasets(gran=gran, target=target)
            train_data, test_data, train_labels, test_labels = custom_train_test_split(data_dict, data, labels)
            best_estimator, best_f1 = cross_validation(train_data, train_labels, gran)
            model_name = f"{dirname}/data/term_freq_models/best_{target}_{gran}.joblib"
            dump(best_estimator, model_name) # save best performing model -> to load -> estimator = load("your-model.joblib")
            predictions = best_estimator.predict(test_data)
            prec = precision_score(test_labels, predictions)
            recall = recall_score(test_labels, predictions)
            f1 = f1_score(test_labels, predictions)
            performance_log.append([target, gran, prec, recall, f1])
    print(performance_log)
    log_file_loc = f"{dirname}/data/term_freq_models/log_performance.pickle"
    with open(log_file_loc, 'wb') as handle:
        pickle.dump(performance_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
