import os
import numpy as np
import pandas as pd
import pickle

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

# Model takes in section location, number of sections, mention of previous primary section, and location percentage
def normalize_section(paper_dict):
    # Input: dictionary containing paper headings
    # Output: list of normalized heading strings
    rev_normalized_sections2numeric = {0: 'abstract', 1: 'introduction', 2: 'methods', 3: 'results', 4: 'discussion', 5: 'conclusions', 6: 'misc.'}

    dirname = os.path.dirname(__file__).rsplit('/', 2)[0]  
    model_loc = os.path.join(dirname, 'data/normalization_model.sav')
    loaded_model = pickle.load(open(model_loc, 'rb'))

    section_list = list(paper_dict.keys())
    length = len(section_list)
    num_sec = [length] * length
    loc = [section_list.index(x) + 1 for x in section_list]
    loc_percentage = [i/j for i, j in zip(loc, num_sec)]


    data = pd.DataFrame(list(zip(section_list, loc, num_sec, loc_percentage)), \
        columns =['section', 'section_loc', 'number_of_sections', 'loc_percentage'])
    
    data['primary_section'] = data.apply(last_primary_section, axis=1)
    data['primary_section'].fillna(method='ffill', inplace=True)
    
    data = data.replace(regex=['[a-zA-Z]'], value=0)
    data = data[['section_loc', 'number_of_sections', 'primary_section', 'loc_percentage']]

    y_pred = loaded_model.predict(data)
    normalized_sections = [rev_normalized_sections2numeric.get(j,j) for j in y_pred]

    # Update to normalized sections and consolidate dictionary entries
    count = 0
    normalized_dict = {}
    for k, v in paper_dict.items():
        v_list = []
        for li in v:
            if type(v) != str:
                v = str(v)
                v_list.append(v)
        if k in ['all_tables', 'all_figure_descriptions']:
            normalized_dict[k] = v
        elif k == "Abstract_sup":
            normalized_dict['abstract'] = " ".join(v_list)
        elif normalized_sections[count] in normalized_dict:
            normalized_dict[normalized_sections[count]] = normalized_dict[normalized_sections[count]] + " " + " ".join(v_list)
        else:
            normalized_dict[normalized_sections[count]] = " ".join(v_list)
        count += 1

    return normalized_dict
