# prediabetes_doc_classifier
This relates to work involving prediabetes classification on biomedical manuscripts using annotations from prior meta-analyses. All code and data necessary to replicate model development is provided. 

Initially, PDFs for every article mentioned within either meta-analysis were converted to xml using GROBID. The PDF and XML output files are both provided. To generate your own XML files using the provided PDFs, please follow the GROBID batch tutorial (https://grobid.readthedocs.io/en/latest/Grobid-batch/). After, run the preprocessing/xml2txt.py file, which generates text files supplemented with abstracts extracted from PubMed. A CSV containing these abstracts is provided. This will need to be generated for new articles. 

To generate labels for comparison, run create_labels.py, which will create both strict and lenient label sets. The strict label set only uses labels when they are annotated in both meta-analyses. The lenient label set uses labels when they are annotated in either meta-analysis. Lenient labels are primarily used in this analysis.

For the term frequency models, train the section normalizer using preprocessing/train_normalize_sections.py. This creates a model that normalizes all manuscript headings (i.e., introduction, methods, results, etc.). Then, run the document_classifier/extract_data_sections.py and document_classifier/extract_data_whole_paper.py to generate term frequency feature data normalized by section. These files contain the regex patterns for each entity. Models using these feature sets can be trained using document_classifier/train_document_classifier.py.

For deep learning models, ...
