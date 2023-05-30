import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score


header = ['gold_igt', 'gold_ifg_ada', 'gold_ifg_who', 'gold_hba1c_ada', 'gold_hba1c_iec']

def generate_labels(row, label_list):
    labels = [0] * len(label_list)
    for j, pattern in enumerate(label_list):
        if pattern in row['Prediabetes prevalence (%)']:
            labels[j] = 1
    
    return labels

def load_data(file_path, label_list, col2index = 'Study'):
    labels = pd.read_csv(file_path)
    labels.rename(columns={"Prediabetes definition": "Prediabetes prevalence (%)",}, inplace=True)
    labels.set_index(col2index, inplace=True)

    labels['labels'] = labels.apply(generate_labels, args=(label_list,), axis=1)
    labels = labels[['labels']]

    labels[header] = pd.DataFrame(labels['labels'].tolist(), index=labels.index)
    labels.drop(columns=['labels'], inplace=True)
    labels.reset_index(inplace=True)
    
    return labels

### Load in annotation data separately ###
cai_in = 'cai_annotations.csv'
cai_lol = ['IGT', 'IFG-ADA', 'IFG-WHO', 'HbA1c-ADA', 'HbA1c-IEC']
cai_labels = load_data(cai_in, cai_lol)

gujral_in = 'gujral_annotations.csv'
gujral_lol = ['IGT', 'IFG (ADA)', 'IFG (WHO)', 'HbA1c (ADA)', 'HbA1c (IEC)']
gujral_labels = load_data(gujral_in, gujral_lol)

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

print("Cai vs. Gujral")
print_cohen_kappa(cai_labels, gujral_labels)
