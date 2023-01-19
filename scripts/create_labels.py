import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Root directory of project
dirname = os.path.dirname(__file__).rsplit('/', 1)[0]

header = ['gold_igt', 'gold_ifg_ada', 'gold_ifg_who', 'gold_hba1c_ada', 'gold_hba1c_iec']

def generate_labels(row, label_list):
    labels = [0] * len(label_list)
    for j, pattern in enumerate(label_list):
        if pattern in row['Prediabetes prevalence (%)']:
            labels[j] = 1
    
    return labels

def load_data(file_path, file_path_out, label_list, col2index = 'Study'):
    labels = pd.read_csv(file_path)
    labels.rename(columns={"Prediabetes definition": "Prediabetes prevalence (%)",}, inplace=True)
    labels.set_index(col2index, inplace=True)

    labels['labels'] = labels.apply(generate_labels, args=(label_list,), axis=1)
    labels = labels[['labels']]

    labels[header] = pd.DataFrame(labels['labels'].tolist(), index=labels.index)
    labels.drop(columns=['labels'], inplace=True)
    labels.to_csv(file_path_out)
    labels.reset_index(inplace=True)
    
    return labels

### Load in Cai and Gujral data separately ###
### Labels were manually normalized; e.g., IFG (ADA, WHO) -> IFG (ADA), IFG (WHO) ###
cai_in = os.path.join(dirname, 'data/cai/annotations/cai_Suppl4_1StudyPerRow.csv')
cai_out = os.path.join(dirname, 'data/cai/annotations/cai_labels.csv')
cai_lol = ['IGT', 'IFG-ADA', 'IFG-WHO', 'HbA1c-ADA', 'HbA1c-IEC']
cai_labels = load_data(cai_in, cai_out, cai_lol)

gujral_in = os.path.join(dirname, 'data/gujral/annotations/gujral_Suppl2_annotations.csv')
gujral_out = os.path.join(dirname, 'data/gujral/annotations/gujral_labels.csv')
gujral_lol = ['IGT', 'IFG (ADA)', 'IFG (WHO)', 'HbA1c (ADA)', 'HbA1c (IEC)']
gujral_labels = load_data(gujral_in, gujral_out, gujral_lol)


### Cohen's Kappa calculations ###
def print_cohen_kappa(labels_1, labels_2):
    scores = []
    merged_labels = labels_1.merge(labels_2, how='inner', on='Study')
    merged_labels.dropna(inplace=True)
    for head in header:
        head_x = head + '_x'
        head_y = head + '_y'
        cks = cohen_kappa_score(merged_labels[head_x], merged_labels[head_y])
        print(f'{head}: {cks}')
        scores.append(cks)
    print(f'Average: {np.mean(scores)}')

print_cohen_kappa(cai_labels, gujral_labels)

### Merge labels from previous datasets ###
### strict labels are those that exactly match (i.e., only keep labels that match) ###
### lenient labels are those that don't exactly match (i.e., some disagreement) ###
def compare_curation_strict(row):
    labels = [0] * len(header)

    for j, head in enumerate(header):
        head_x = head + '_x'
        head_y = head + '_y'

        if row[head_x] == row[head_y]:
            labels[j] = row[head_x]
        else:
            labels[j] = None
    
    return labels

def compare_curation_lenient(row):
    labels = [0] * len(header)

    for j, head in enumerate(header):
        head_x = head + '_x'
        head_y = head + '_y'

        if row[head_x] == 1 or row[head_y] == 1:
            labels[j] = 1
        else:
            labels[j] = 0
    
    return labels

def merge_labels(labels_1, labels_2, method, out_file):
    if method.lower() == 'strict':
        how = 'inner'
        comparison = compare_curation_strict
    else:
        how = 'outer'
        comparison = compare_curation_lenient

    merged_labels = labels_1.merge(labels_2, how=how, on='Study')
    merged_labels.set_index('Study', inplace=True)

    merged_labels['labels'] = merged_labels.apply(comparison, axis=1)
    merged_labels = merged_labels[['labels']]
    

    merged_labels[header] = pd.DataFrame(merged_labels['labels'].tolist(), index=merged_labels.index)
    merged_labels.drop(columns='labels', inplace=True)
    merged_labels.dropna(inplace=True)

    merged_labels.to_csv(out_file)

    return merged_labels

### Strict ###
strict_out = os.path.join(dirname, 'data/labels_strict.csv')
strict_labels = merge_labels(cai_labels, gujral_labels, 'strict', strict_out)
print(strict_labels.head()) # sanity check

### Lenient ###
lenient_out = os.path.join(dirname, 'data/labels_lenient.csv')
lenient_labels = merge_labels(cai_labels, gujral_labels, 'lenient', lenient_out)
print(lenient_labels.head()) # sanity check
