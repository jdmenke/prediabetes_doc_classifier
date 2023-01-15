import os
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
import seaborn as sns

import pickle


### This file normalizes sections extracted by GROBID into the following sections: 
### abstract, introduction, methods, results, discussion, conclusion, misc.

## Workflow: 
#           1) Create vector for extracted heading (simple average of lowercased words)
#           2) Feed into decision tree (inputs: section, number of headings, location of current heading, location ratio, previous terms normalzied heading)

dirname = os.path.dirname(__file__).rsplit('/', 2)[0]
data_raw = pd.read_csv(os.path.join(dirname, 'data/normalized_section_training_data.csv'))  

data = data_raw[(data_raw['normalized_section'] != 'tables') & (data_raw['normalized_section'] != 'figures')].reset_index(drop=True)
section_list = data.groupby('paper')['section'].apply(list)
normalized_section_list = data.groupby('paper')['normalized_section'].apply(list)

data['section_list'] = data['paper'].map(section_list)
data['normalized_section_list'] = data['paper'].map(normalized_section_list)

def section_location(row):
    output = row['section_list'].index(row['section']) + 1
    return output

data['section_loc'] = data.apply(section_location, axis=1)
data['number_of_sections'] = data['section_list'].apply(lambda ls: len(ls))

numeric_paper = list(set(data['paper'].tolist()))
numeric_paper.sort()
paper2numeric = {}
for idx, nls in enumerate(numeric_paper):
    paper2numeric[nls] = idx

## Section Encoding ##
# Set previous_ns as 0 when starting a paper, then replace with value from normalized sections
normalized_sections2numeric = {'abstract': 0, 'intro': 1, 'methods': 2, 'results': 3, 'discussion': 4, 'conclusions': 5, 'misc.': 6}
rev_normalized_sections2numeric = {0: 'abstract', 1: 'introduction', 2: 'methods', 3: 'results', 4: 'discussion', 5: 'conclusions', 6: 'misc.'}

def last_primary_section(row):
    output = np.nan
    if 'abstract' in row['section'].lower().replace(' ', '').replace('-', '').strip():
        output = 0
    elif 'introduct' in row['section'].lower().replace(' ', '').replace('-', '').strip():
        output = 1
    elif ('method' in row['section'].lower().replace(' ', '').replace('-', '').strip()) or ('materials' in row['section'].lower().replace(' ', '').replace('-', '').strip()):
        output = 2
    elif 'result' in row['section'].lower().replace(' ', '').replace('-', '').strip():
        output = 3
    elif 'discussion' in row['section'].lower().replace(' ', '').replace('-', '').strip():
        output = 4
    elif 'conclusion' in row['section'].lower().replace(' ', '').replace('-', '').strip():
        output = 5
    elif ('awknow' in row['section'].lower().replace(' ', '').replace('-', '').strip()) or ('conflictofinterest' in row['section'].lower().replace(' ', '').replace('-', '').strip()):
        output = 6
    return output

data['primary_section'] = data.apply(last_primary_section, axis=1)
data['primary_section'].fillna(method='ffill', inplace=True)

data['paper'] = data['paper'].replace(paper2numeric)
data['section'] = data['section'].str.lower()

data['normalized_section'] = data['normalized_section'].replace(normalized_sections2numeric)
data['loc_percentage'] = data['section_loc'] / data['number_of_sections']
data = data.drop(['section', 'section_list', 'normalized_section_list'], axis=1)

### Models ###
## Scale data ##
X = data.drop('normalized_section', axis=1)
y = data["normalized_section"]
grouped_by_paper_list = np.array(data['paper'].values)
X = X.drop('paper', axis=1)

#### Determining which classifier to pick ####
## Splitting training and testing sets by paper ##
gkf = list(GroupKFold(n_splits=10).split(X, y, grouped_by_paper_list))
scoring = {'F1': make_scorer(f1_score, average='weighted', zero_division=1), 'Precision': make_scorer(precision_score, average='weighted', zero_division=1), 'Recall': make_scorer(recall_score, average='weighted', zero_division=1)}

def collect_results(res):
    Pm = res.cv_results_['mean_test_Precision'][res.best_index_]
    Pstd = res.cv_results_['std_test_Precision'][res.best_index_]
    prec = (Pm, Pstd)

    Rm = res.cv_results_['mean_test_Recall'][res.best_index_]
    Rstd = res.cv_results_['std_test_Recall'][res.best_index_]
    recall = (Rm, Rstd)

    F1m = res.cv_results_['mean_test_F1'][res.best_index_]
    F1std = res.cv_results_['std_test_F1'][res.best_index_]
    F1 = (F1m, F1std)

    return prec, recall, F1

## SVM ##
clf = SVC(random_state=42)
param_grid = {
    'C': [0.5, 1, 2, 10],
    'kernel': ['linear', 'rbf'],
    'tol': [1e-2, 1e-3, 1e-4],
}
grid_search = GridSearchCV(clf, param_grid, cv=gkf, refit='F1', scoring=scoring, verbose=3, n_jobs=-1)
result = grid_search.fit(X, y)

svm_be = result.best_estimator_
svm_precision, svm_recall, svm_f1 = collect_results(result)

## KNN ##
clf = KNeighborsClassifier()
param_grid = {
    'n_neighbors': list(range(1, 20)),
}
grid_search = GridSearchCV(clf, param_grid, cv=gkf, refit='F1', scoring=scoring, verbose=3, n_jobs=-1)
result = grid_search.fit(X, y)

knn_be = result.best_estimator_
knn_precision, knn_recall, knn_f1 = collect_results(result)

## Decision Tree ##
clf = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': list(range(1, 5)),
    'min_samples_split': list(range(2, 15)),
    'min_samples_leaf': list(range(1, 15)),
}
grid_search = GridSearchCV(clf, param_grid, cv=gkf, refit='F1', scoring=scoring, verbose=3, n_jobs=-1)
result = grid_search.fit(X, y)

dt_be = result.best_estimator_

## Performance ##
dt_precision, dt_recall, dt_f1 = collect_results(result)

# plot_tree(result.best_estimator_)
# plt.figure(figsize=(24,24))
# plt.show()

## Random Forest ##
clf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': list(range(1, 100, 25)),
    'max_depth': [None, 1, 2, 3, 4, 5],
    'min_samples_leaf': list(range(1, 15)),
}
grid_search = GridSearchCV(clf, param_grid, cv=gkf, refit='F1', scoring=scoring, verbose=3, n_jobs=-1)
result = grid_search.fit(X, y)

rf_be = result.best_estimator_
rf_precision, rf_recall, rf_f1 = collect_results(result)

## XgBoost ##
clf = xgb.XGBClassifier()
param_grid = {
    'n_estimators': list(range(1, 100, 25)),
    'max_depth': [None, 1, 2, 3, 4, 5],
    'max_leaves': list(range(1, 15)),
}
grid_search = GridSearchCV(clf, param_grid, cv=gkf, refit='F1', scoring=scoring, verbose=3, n_jobs=-1)
result = grid_search.fit(X, y)

xgb_be = result.best_estimator_
xgb_precision, xgb_recall, xgb_f1 = collect_results(result)

## Performances ##
print(f"Precision: {svm_precision[0]} +/- {svm_precision[1]}")
print(f"Recall: {svm_recall[0]} +/- {svm_recall[1]}")
print(f"F1: {svm_f1[0]} +/- {svm_f1[1]}")

print(f"Precision: {knn_precision[0]} +/- {knn_precision[1]}")
print(f"Recall: {knn_recall[0]} +/- {knn_recall[1]}")
print(f"F1: {knn_f1[0]} +/- {knn_f1[1]}")

print(f"Precision: {dt_precision[0]} +/- {dt_precision[1]}")
print(f"Recall: {dt_recall[0]} +/- {dt_recall[1]}")
print(f"F1: {dt_f1[0]} +/- {dt_f1[1]}")

print(f"Precision: {rf_precision[0]} +/- {rf_precision[1]}")
print(f"Recall: {rf_recall[0]} +/- {rf_recall[1]}")
print(f"F1: {rf_f1[0]} +/- {rf_f1[1]}")

print(f"Precision: {xgb_precision[0]} +/- {xgb_precision[1]}")
print(f"Recall: {xgb_recall[0]} +/- {xgb_recall[1]}")
print(f"F1: {xgb_f1[0]} +/- {xgb_f1[1]}")

### Select and save best performing model ###
overall_f1 = dt_f1[0]
best_model = dt_be
if svm_f1[0] > overall_f1:
    overall_f1 = svm_f1[0]
    best_model = svm_be
elif knn_f1[0] > overall_f1:
    overall_f1 = knn_f1[0]
    best_model = knn_be
elif rf_f1[0] > overall_f1:
    overall_f1 = rf_f1[0]
    best_model = rf_be
elif xgb_f1[0] > overall_f1:
    overall_f1 = xgb_f1[0]
    best_model = xgb_be

print(best_model)
model_loc = os.path.join(dirname, 'data/normalization_model.sav')
pickle.dump(best_model, open(model_loc, 'wb'))