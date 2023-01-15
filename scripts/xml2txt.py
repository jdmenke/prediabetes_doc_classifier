from bs4 import BeautifulSoup
import json
import os
from collections import defaultdict
import pandas as pd

def xml2text(input):
        # Opens XML file and reads into soup using BeautifulSoup module
        with open(input) as fp:
                contents = fp.read()
                soup = BeautifulSoup(contents,'xml')
                
                # Initialize dictionary where we will store paper text by section
                paper_dictionary = defaultdict(list)
                
                # Abstract Section
                try:
                        abstract = soup.find('abstract')
                        abstract_content = abstract.findChildren(recursive=False)
                        full_abstract_text = ''
                        for ab_text in abstract_content:
                                full_abstract_text = full_abstract_text + ' ' + ab_text.get_text()
                        paper_dictionary['Abstract'] = [full_abstract_text.strip()]
                except:
                        pass
                
                #print(input)   # For debugging

                # Body Section (Intro, Methods, Results, Discussion, Conclusion, Misc.)
                body = soup.find('body')
                sections = body.find_all('head')
                headings_raw = []
                for section in sections:
                        headings_raw.append(section.get_text())

                for item in headings_raw:
                        heading_text = body.find_all("head", string=item)
                        for head in heading_text:
                                result = list(head.next_siblings)
                                section_text = []
                                for i in range(len(result)):
                                        text = result[i].get_text()
                                        section_text.append(text)
                                if paper_dictionary[item]:
                                        paper_dictionary[item].append(section_text)
                                else:
                                        paper_dictionary[item] = section_text
                
                # Creates a list of lists wherein each record represents 1 table containing 3 items (table name, table description [if available], and structured table)
                root_table = []
                figures = []
                test = soup.find_all("figure")
                for fig in test:
                    tables = []
                    if fig.get('type') == 'table':
                        paper_dictionary.pop((fig.find('head')).get_text(), 'KeyErrorPrevention')
                        tables.append(str((fig.find('head')).get_text()))
                        tables.append(((fig.find('figDesc')).get_text()))
                        tables.append(str(fig.find('table')))
                    else:
                        paper_dictionary.pop((fig.find('head')).get_text(), 'KeyErrorPrevention')
                        figures.append(str(fig.find('figDesc').get_text()))
                    root_table.append(tables)
                
                paper_dictionary['all_tables'] = root_table
                paper_dictionary['all_figure_descriptions'] = figures

        return paper_dictionary


def directory_looper(directory):
        input_list=[]
        base_name_list=[]
        output_list=[]

        # TODO: make more efficient
        for name in os.listdir(directory):
                filename = os.fsdecode(name)
                if filename.endswith(".tei.xml"):
                        input_list.append(os.path.join(directory, filename))
        
        for name in input_list:
                base_name = name.rsplit('/', 1)[-1].split('.')[0]
                if base_name == "Wang_2007_2":
                        base_name = "Wang 2007_2"
                else:
                        base_name = base_name.replace("_", " ")
                base_name_list.append(base_name)

        for name in input_list:
                size = len(name)
                base_name = name[:size - 8]
                out_name = base_name.replace('/xml/', '/txt/')
                output_list.append(out_name + ".txt")
        
        return input_list, base_name_list, output_list


def process_directory(directory, abstract_input):
        inputs, base, outputs = directory_looper(directory)
        for j in range(len(inputs)):
                paper_dict = xml2text(inputs[j]) # converts xml to txt
                
                # Supplement abstracts
                query_statement = f"id == '{base[j]}'"
                intermed = abstract_input.query(query_statement)
                intermed.set_index('id', inplace=True)

                list2add = []
                add_abstract = intermed.at[base[j], 'abstract']
                list2add.append(add_abstract)
                paper_dict['Abstract_sup'] = list2add

                with open(outputs[j], 'w') as fw:
                        fw.write(json.dumps(paper_dict))


if __name__ == "__main__":
        dirname = os.path.dirname(__file__).rsplit('/', 2)[0]                                   # root directory
        abstract_input = pd.read_csv(os.path.join(dirname, 'data/abstracts_journals.csv'))      # csv of pmid, id (e.g., Kim 2018), journal, abstract 
        process_directory(os.path.join(dirname, 'data/cai/xml'), abstract_input)                # Cai papers
        process_directory(os.path.join(dirname, 'data/gujral/xml'), abstract_input)             # Gujral papers