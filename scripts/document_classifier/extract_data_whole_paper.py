from bs4 import BeautifulSoup
import json
import csv
import os
import re
import pandas as pd
import numpy as np
from run_normalize_sections import normalize_section
from nltk.tokenize import sent_tokenize

#### Global Variables ###
# a_ for abstract; i_ for introduction; m_ for methods; r_ for results; d_ for discussion; c_ for conclusion; t_ for table
name_modifiers = ['', '_ref', '_add']
entities = ['IFG', 'IGT', 'HbA1c', 'WHO', 'ADA', 'IEC', 'IGT_WHO', 'IGT_ADA', 'IFG_ADA', 'IFG_WHO', 'HbA1c_ADA', 'HbA1c_IEC']
header_original = ['paper', 'year']
header = header_original.copy()

for en in entities:
        for nm in name_modifiers:
                col_name = f"{en}{nm}"
                header.append(col_name)

### Functions ###
def processpaper(input, base_name):
        print(base_name)
        pub_year = re.search("\d{4}", base_name).group(0)
        whole_paper = [base_name, int(pub_year)]
        abstract_data = [0] * (len(header) - len(header_original))
        intro_data = [0] * (len(header) - len(header_original))
        methods_data = [0] * (len(header) - len(header_original))
        results_data = [0] * (len(header) - len(header_original))
        discussion_data = [0] * (len(header) - len(header_original))
        conclusion_data = [0] * (len(header) - len(header_original))

        with open(input) as fp:                
                contents = fp.read()
                paper_dict = json.loads(contents)
                table_text = paper_dict['all_tables']
                table_data = search4patterns(str(table_text))
                normalized_dict = normalize_section(paper_dict)
        

        for section in normalized_dict.keys():
                # skip miscellaneous sections like awknowledgements, conflict of interest, references, etc.
                if section in ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusions']:
                        section_content = normalized_dict[section]
                        if section in 'abstract':
                                abstract_data = search4patterns(section_content)
                        elif section in 'introduction':
                                intro_data = search4patterns(section_content)
                        elif section in 'methods':
                                methods_data = search4patterns(section_content)
                        elif section in 'results':
                                results_data = search4patterns(section_content)
                        elif section in 'discussion':
                                discussion_data = search4patterns(section_content)
                        elif section in 'conclusion':
                                conclusion_data = search4patterns(section_content)
                else:
                        pass
        
        whole_paper_ = np.array([table_data, abstract_data, intro_data, methods_data, results_data, discussion_data, conclusion_data])
        whole_paper_data = whole_paper_.sum(axis=0)
        whole_paper.extend(whole_paper_data)

        return [whole_paper]


def search4patterns(section_text=str):
        #### Patterns for organizations and diagnosis criteria ####
        between_words = "[A-Za-z\d\-\s]"
        ### Organizations ###
        ## WHO ##
        WHO_lng = "[Ww]orld\s[Hh]ealth\s[Oo]rganization"
        WHO_abbr = "WHO"
        WHO_whole = "[Ww]orld\s[Hh]ealth\s[Oo]rganization\s\(WHO\)"
        WHO_regex_pattern_abbr = rf"""
                {WHO_whole}\scriteria|{WHO_whole}\sdefinitions?|{WHO_whole}\sclassifications?|{WHO_whole}\srecommendations?|{WHO_whole}\sthresholds?|{WHO_whole}\scategories|
                {WHO_abbr}\scriteria|{WHO_abbr}\sdefinitions?|{WHO_abbr}\sclassifications?|{WHO_abbr}\srecommendations?|{WHO_abbr}\sthresholds?|{WHO_abbr}\scategories|
                {WHO_lng}\scriteria|{WHO_lng}\sdefinitions?|{WHO_lng}\sclassifications?|{WHO_lng}\srecommendations?|{WHO_lng}\sthresholds?|{WHO_lng}\scategories|
                {WHO_whole}{between_words}+criteria|{WHO_whole}{between_words}+definitions?|{WHO_whole}{between_words}+classifications?|{WHO_whole}{between_words}+recommendations?|{WHO_whole}{between_words}+thresholds?|{WHO_whole}{between_words}+categories|
                {WHO_abbr}{between_words}+criteria|{WHO_abbr}{between_words}+definitions?|{WHO_abbr}{between_words}+classifications?|{WHO_abbr}{between_words}+recommendations?|{WHO_abbr}{between_words}+thresholds?|{WHO_abbr}{between_words}+categories|
                {WHO_lng}{between_words}+criteria|{WHO_lng}{between_words}+definitions?|{WHO_lng}{between_words}+classifications?|{WHO_lng}{between_words}+recommendations?|{WHO_lng}{between_words}+thresholds?|{WHO_lng}{between_words}+categories|
                {WHO_lng}|6\.1\s*(?:to|and|\-)\s*6\.9\s*mmol/[Ll]|110(?:\.\d+)?\s*(?:to|and|\-|\–)\s*(?:124(?:\.\d+)?|125(?:\.\d+)?)\s*mg/d[Ll]|
                7\.8\s*\-\s*11\.0\s*mmol/[Ll]|7\.8\s*to\s*11\.0\s*mmol/[Ll]|(?:140(?:\.\d+)?|141(?:\.\d+)?)\s*\-\s*(?:198(?:\.\d+)?|199(?:\.\d+)?|200(?:\.\d+)?)\s*mg/d[Ll]|(?:140(?:\.\d+)?|141(?:\.\d+)?)\s*to\s*(?:198(?:\.\d+)?|199(?:\.\d+)?|200(?:\.\d+)?)\s*mg/d[Ll]"""
        ## ADA ##
        ADA_lng = "[Aa]merican\s[Dd]iabetes\s[Aa]ssociation"
        ADA_abbr = "ADA"
        ADA_whole = "[Aa]merican\s[Dd]iabetes\s[Aa]ssociation\s\(ADA\)"
        ADA_regex_pattern_abbr = rf"""
                {ADA_whole}\scriteria|{ADA_whole}\sdefinitions?|{ADA_whole}\sclassifications?|{ADA_whole}\srecommendations?|{ADA_whole}\sthresholds?|{ADA_whole}\scategories|
                {ADA_abbr}\scriteria|{ADA_abbr}\sdefinitions?|{ADA_abbr}\sclassifications?|{ADA_abbr}\srecommendations?|{ADA_abbr}\sthresholds?|{ADA_abbr}\scategories|
                {ADA_lng}\scriteria|{ADA_lng}\sdefinitions?|{ADA_lng}\sclassifications?|{ADA_lng}\srecommendations?|{ADA_lng}\sthresholds?|{ADA_lng}\scategories|
                {ADA_whole}{between_words}+criteria|{ADA_whole}{between_words}+definitions?|{ADA_whole}{between_words}+classifications?|{ADA_whole}{between_words}+recommendations?|{ADA_whole}{between_words}+thresholds?|{ADA_whole}{between_words}+categories|
                {ADA_abbr}{between_words}+criteria|{ADA_abbr}{between_words}+definitions?|{ADA_abbr}{between_words}+classifications?|{ADA_abbr}{between_words}+recommendations?|{ADA_abbr}{between_words}+thresholds?|{ADA_abbr}{between_words}+categories|
                {ADA_lng}{between_words}+criteria|{ADA_lng}{between_words}+definitions?|{ADA_lng}{between_words}+classifications?|{ADA_lng}{between_words}+recommendations?|{ADA_lng}{between_words}+thresholds?|{ADA_lng}{between_words}+categories|
                {ADA_lng}|{ADA_abbr}\s\(.*Diabetes.*\)|7\.8\s*(?:to|and|\-|\–)\s*11(?:\.0)?\s*mmol/[Ll]|
                (?:140(?:\.\d+)?|141(?:\.\d+)?)\s*\-\s*(?:198(?:\.\d+)?|199(?:\.\d+)?|200(?:\.\d+)?)\s*mg/d[Ll]|(?:140(?:\.\d+)?|141(?:\.\d+)?)\s*to\s*(?:198(?:\.\d+)?|199(?:\.\d+)?|200(?:\.\d+)?)\s*mg/d[Ll]|
                5\.6\s*(?:to|and|\-|\–)\s*6\.9\s*mmol/[Ll]|(?:100|101)\s*(?:to|and|\-|\–)\s*(?:124(?:\.\d+)?|125(?:\.\d+)?)\s*mg\/d[Ll]|
                39.{0,2}(?:and|to|\-|\–).{0,2}(47(?:\.\d+)?|48(?:\.\d+)?)mmol\s?/\s?mol|5\.7.{0,2}(?:and|to|\-|\–).{0,2}6\.[45].{0,2}([Pp]ercent|\%)"""
        ## IEC ##
        IEC_lng = "[Ii]nternational\s[Ee]xpert\s[Cc]ommittee"
        IEC_abbr = "IEC"
        IEC_whole = "[Ii]nternational\s[Ee]xpert\s[Cc]ommittee\s\(IEC\)"
        IEC_regex_pattern_abbr = rf"""
                {IEC_whole}\scriteria|{IEC_whole}\sdefinitions?|{IEC_whole}\sclassifications?|{IEC_whole}\srecommendations?|{IEC_whole}\sthresholds?|{IEC_whole}\scategories|
                {IEC_abbr}\scriteria|{IEC_abbr}\sdefinitions?|{IEC_abbr}\sclassifications?|{IEC_abbr}\srecommendations?|{IEC_abbr}\sthresholds?|{IEC_abbr}\scategories|
                {IEC_lng}\scriteria|{IEC_lng}\sdefinitions?|{IEC_lng}\sclassifications?|{IEC_lng}\srecommendations?|{IEC_lng}\sthresholds?|{IEC_lng}\scategories|
                {IEC_whole}{between_words}+criteria|{IEC_whole}{between_words}+definitions?|{IEC_whole}{between_words}+classifications?|{IEC_whole}{between_words}+recommendations?|{IEC_whole}{between_words}+thresholds?|{IEC_whole}{between_words}+categories|
                {IEC_abbr}{between_words}+criteria|{IEC_abbr}{between_words}+definitions?|{IEC_abbr}{between_words}+classifications?|{IEC_abbr}{between_words}+recommendations?|{IEC_abbr}{between_words}+thresholds?|{IEC_abbr}{between_words}+categories|
                {IEC_lng}{between_words}+criteria|{IEC_lng}{between_words}+definitions?|{IEC_lng}{between_words}+classifications?|{IEC_lng}{between_words}+recommendations?|{IEC_lng}{between_words}+thresholds?|{IEC_lng}{between_words}+categories|
                {IEC_lng}|42\s*(?:to|and|\-|\–)\s*47\s*mmol/mol|6\.0\%\s*(?:to|and|\-|\–)\s*6\.4\%|6\.0\s*[Pp]ercent\s*(?:to|and|\-|\–)\s*6\.4\s*[Pp]ercent"""

        WHO_pattern = re.compile(WHO_regex_pattern_abbr, re.VERBOSE)
        ADA_pattern = re.compile(ADA_regex_pattern_abbr, re.VERBOSE)
        IEC_pattern = re.compile(IEC_regex_pattern_abbr, re.VERBOSE)
        
        ### Diagnosis Criteria ###
        IFG_pattern = re.compile(r"""IFG|\s+ifg\s+|[Ii]mpaired\s[Ff]asting\s[Gg]|FPG|[Ii]mpaired\sFBG|[Ee]levated\s[Ff]asting\s[Gg]l|[Ff]asting\s(?:(?:[Ww]hole)?\s*[Bb]lood|[Ss]erum|[Pp]lasma)\s[Gg]l|[Ff]asting\s[Gg]lycemia""", re.VERBOSE)
        IGT_pattern = re.compile(r"""IGT|\s+igt\s+|[Ii]mpaired\sglucose\stolerance|[Gg]lucose\s[Tt]olerance|OGTT|[Oo]ral\s[Gg]lucose\s[Tt]olerance\s[Tt]est|2hPG|2(?:\-|\–)h(?:our)?\s(?:[Pp]lasma\s)?[Gg]lucose""", re.VERBOSE)
        HbA1c_pattern = re.compile(r"""HbA\(1c\)\-defined\sprediabetes|HbA1c|hemoglobin\sA1c|HbA\(1c\)|HbA1\sC|HbA\s1c|[Gg]lycosylated\s[Hh]emoglobin|HgbA1c|[Gg]lycated\s[Hh]emoglobin|[Ee]levated\sA1[Cc]""", re.VERBOSE)

        ### Sentence references another study ###
        ref_pattern = re.compile(r"""(?:[\[\(]\d{1,3}(?:[\,-]\s?\d{1,3})*[\]\)])+[\.,\,]|[\.,\,]\s?(?:[\[\(]\d{1,3}[\]\)])+|et\sal""", re.VERBOSE)

        ### Additional ###
        additional_pattern = re.compile(r"""\sdetermin|\sclassif|\sdefin|\scategor|\sconsid|\saccord|\sdivid|\s[Cc]riter""", re.VERBOSE)

        ### Secondary Patterns ###
        IFG_ADA_secondary_pattern = re.compile(r"""5\.[567]""", re.VERBOSE)
        IFG_WHO_secondary_pattern = re.compile(r"""6\.[012]""", re.VERBOSE)
        HbA1c_ADA_secondary_pattern = re.compile(r"""(?:38|39|40)|5\.[678]""", re.VERBOSE)
        HbA1c_IEC_secondary_pattern = re.compile(r"""(?:41|42|43)|6\.[01]""", re.VERBOSE)

        # Run patterns across whole section
        WHO_mention = WHO_pattern.findall(section_text)
        ADA_mention = ADA_pattern.findall(section_text)
        IEC_mention = IEC_pattern.findall(section_text)
        IFG_mention = IFG_pattern.findall(section_text)
        IGT_mention = IGT_pattern.findall(section_text)
        HbA1c_mention = HbA1c_pattern.findall(section_text)

        # Within sentences of a section
        doc = sent_tokenize(section_text)
        # Prediabetes diagnosis combinations
        IGT_WHO = 0
        IGT_ADA = 0
        IFG_ADA = 0
        IFG_WHO = 0
        HbA1c_ADA = 0
        HbA1c_IEC = 0
        # Reference and additional patterns combinations
        IGT_ref = 0
        IGT_add = 0
        IFG_ref = 0
        IFG_add = 0
        HbA1c_ref = 0
        HbA1c_add = 0
        WHO_ref = 0
        WHO_add = 0
        ADA_ref = 0
        ADA_add = 0
        IEC_ref = 0
        IEC_add = 0
        # Additional combinations
        IGT_WHO_ref = 0
        IGT_WHO_add = 0
        IGT_ADA_ref = 0
        IGT_ADA_add = 0
        IFG_ADA_ref = 0
        IFG_ADA_add = 0
        IFG_WHO_ref = 0
        IFG_WHO_add = 0
        HbA1c_ADA_ref = 0
        HbA1c_ADA_add = 0
        HbA1c_IEC_ref = 0
        HbA1c_IEC_add = 0

        section_cache = {}
        for sentence in doc:
                if sentence not in section_cache:
                        # Primary Patterns
                        ref_bool = len(ref_pattern.findall(sentence))
                        add_bool = len(additional_pattern.findall(sentence))
                        WHO_bool = len(WHO_pattern.findall(sentence))
                        ADA_bool = len(ADA_pattern.findall(sentence))
                        IEC_bool = len(IEC_pattern.findall(sentence))
                        IFG_bool = len(IFG_pattern.findall(sentence))
                        IGT_bool = len(IGT_pattern.findall(sentence))
                        HbA1c_bool = len(HbA1c_pattern.findall(sentence))
                        any_positive = [WHO_bool, ADA_bool, IEC_bool, IFG_bool, IGT_bool, HbA1c_bool]

                        # Secondary Patterns
                        if IFG_bool > 0:
                                IFG_ADA_bool = len(IFG_ADA_secondary_pattern.findall(sentence))
                                if IFG_ADA_bool > 0:
                                        IGT_ADA += 1
                                IFG_WHO_bool = len(IFG_WHO_secondary_pattern.findall(sentence))
                                if IFG_WHO_bool > 0:
                                        IFG_WHO += 1
                        if HbA1c_bool > 0:
                                HbA1c_ADA_bool = len(HbA1c_ADA_secondary_pattern.findall(sentence))
                                if HbA1c_ADA_bool > 0:
                                        HbA1c_ADA += 1
                                HbA1c_IEC_bool = len(HbA1c_IEC_secondary_pattern.findall(sentence))
                                if HbA1c_IEC_bool > 0:
                                        HbA1c_IEC += 1
                        # Sentence extraction if criteria detected
                        # Diagnosis combinations
                        if IGT_bool > 0 and WHO_bool > 0:
                                IGT_WHO += 1
                        if IGT_bool > 0 and ADA_bool > 0:
                                IGT_ADA += 1
                        if IFG_bool > 0 and ADA_bool > 0:
                                IFG_ADA += 1
                        if IFG_bool > 0 and WHO_bool > 0:
                                IFG_WHO += 1
                        if HbA1c_bool > 0 and ADA_bool > 0:
                                HbA1c_ADA += 1
                        if HbA1c_bool > 0 and IEC_bool > 0:
                                HbA1c_IEC += 1
                        # Additional - original criteria
                        if IGT_bool > 0 and ref_bool > 0:
                                IGT_ref += 1
                        if IGT_bool > 0 and add_bool > 0:
                                IGT_add += 1
                        if IFG_bool > 0 and ref_bool > 0:
                                IFG_ref += 1
                        if IFG_bool > 0 and add_bool > 0:
                                IFG_add += 1
                        if HbA1c_bool > 0 and ref_bool > 0:
                                HbA1c_ref += 1
                        if HbA1c_bool > 0 and add_bool > 0:
                                HbA1c_add += 1
                        if WHO_bool > 0 and ref_bool > 0:
                                WHO_ref += 1
                        if WHO_bool > 0 and add_bool > 0:
                                WHO_add += 1
                        if ADA_bool > 0 and ref_bool > 0:
                                ADA_ref += 1
                        if ADA_bool > 0 and add_bool > 0:
                                ADA_add += 1
                        if IEC_bool > 0 and ref_bool > 0:
                                IEC_ref += 1
                        if IEC_bool > 0 and add_bool > 0:
                                IEC_add += 1
                        # Additional - combined criteria
                        if IGT_bool > 0 and WHO_bool > 0 and ref_bool > 0:
                                IGT_WHO_ref += 1
                        if IGT_bool > 0 and WHO_bool > 0 and add_bool > 0:
                                IGT_WHO_add += 1
                        if IGT_bool > 0 and ADA_bool > 0 and ref_bool > 0:
                                IGT_ADA_ref += 1
                        if IGT_bool > 0 and ADA_bool > 0 and add_bool > 0:
                                IGT_ADA_add += 1
                        if IFG_bool > 0 and ADA_bool > 0 and ref_bool > 0:
                                IFG_ADA_ref += 1
                        if IFG_bool > 0 and ADA_bool > 0 and add_bool > 0:
                                IFG_ADA_add += 1
                        if IFG_bool > 0 and WHO_bool > 0 and ref_bool > 0:
                                IFG_WHO_ref += 1
                        if IFG_bool > 0 and WHO_bool > 0 and add_bool > 0:
                                IFG_WHO_add += 1
                        if HbA1c_bool > 0 and ADA_bool > 0 and ref_bool > 0:
                                HbA1c_ADA_ref += 1
                        if HbA1c_bool > 0 and ADA_bool > 0 and add_bool > 0:
                                HbA1c_ADA_add += 1
                        if HbA1c_bool > 0 and IEC_bool > 0 and ref_bool > 0:
                                HbA1c_IEC_ref += 1
                        if HbA1c_bool > 0 and IEC_bool > 0 and add_bool > 0:
                                HbA1c_IEC_add += 1
                        section_cache[sentence] = [len(IFG_mention), IFG_ref, IFG_add, len(IGT_mention), IGT_ref, IGT_add, len(HbA1c_mention), HbA1c_ref, HbA1c_add, len(WHO_mention), WHO_ref, WHO_add, len(ADA_mention), ADA_ref, ADA_add, len(IEC_mention), IEC_ref, IEC_add, IGT_WHO, IGT_WHO_ref, IGT_WHO_add, IGT_ADA, IGT_ADA_ref, IGT_ADA_add, IFG_ADA, IFG_ADA_ref, IFG_ADA_add, IFG_WHO, IFG_WHO_ref, IFG_WHO_add, HbA1c_ADA, HbA1c_ADA_ref, HbA1c_ADA_add, HbA1c_IEC, HbA1c_IEC_ref, HbA1c_IEC_add]
                else:
                        pass                

        # Create row containing desired information (pattern counts) to add for each section
        results_list = [len(IFG_mention), IFG_ref, IFG_add, len(IGT_mention), IGT_ref, IGT_add, len(HbA1c_mention), HbA1c_ref, HbA1c_add, len(WHO_mention), WHO_ref, WHO_add, len(ADA_mention), ADA_ref, ADA_add, len(IEC_mention), IEC_ref, IEC_add, IGT_WHO, IGT_WHO_ref, IGT_WHO_add, IGT_ADA, IGT_ADA_ref, IGT_ADA_add, IFG_ADA, IFG_ADA_ref, IFG_ADA_add, IFG_WHO, IFG_WHO_ref, IFG_WHO_add, HbA1c_ADA, HbA1c_ADA_ref, HbA1c_ADA_add, HbA1c_IEC, HbA1c_IEC_ref, HbA1c_IEC_add]

        return results_list



def directory_looper(directory):
        input_list=[]
        base_name_list=[]

        for name in os.listdir(directory):
                filename = os.fsdecode(name)
                if filename.endswith(".txt"):
                        input_list.append(os.path.join(directory, filename))
        
        for name in input_list:
                base_name = name.rsplit('/', 1)[-1]
                base_name = base_name[:-4]
                if base_name == "Wang_2007_2":
                        base_name = "Wang 2007_2"
                else:
                        base_name = base_name.replace("_", " ")
                base_name_list.append(base_name)
        
        return input_list, base_name_list


def process_directory(input_directory, output_directory):
        inputs, base_names = directory_looper(input_directory)
        regex_analysis = os.path.join(output_directory, 'training_testing_data_wp.csv')
        with open(regex_analysis, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(len(inputs)):
                        csv_data = processpaper(inputs[i], base_names[i])
                        writer.writerows(csv_data)


if __name__ == "__main__":
        dirname = os.path.dirname(__file__).rsplit('/', 2)[0]
        process_directory(os.path.join(dirname, 'data/cai/txt'), os.path.join(dirname, 'data/cai')) ## Cai
        process_directory(os.path.join(dirname, 'data/gujral/txt'), os.path.join(dirname, 'data/gujral')) ## Gujral