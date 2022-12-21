# master
https://drive.google.com/drive/folders/1rQr9RNYq3IQk4AXSozF7vpCBW46sI7ti?usp=sharing

##Repository for the master's thesis "Knowledge graph embeddings for semantic relation clustering"

#Abstract
This master's thesis investigates the usage of knowledge graph embedding for semantic relation clustering. The open information extraction corpus (OPIEC) serves...

#Data
OPIEC corpus can be loaded from ...
In my experiments, I used OPIEK-linked-triples

#Code execution

The code requires python 3.7 and not higher. 

To run the code as described in the thesis, please, install dependencies with pip install -r requirements.txt command and then run main.py file

#Script description

main.py - file from which the whole experiment work described in thesis can be replicated.
Required files are:
1. "Gold_standard_manual.csv" since it was uploaded after manual inspection.
2. OPIEC-linked triples stored in data folder or the path to the triple files has to bee inserted in main.py in the line ...

Read_corpus.py
The script read the OPIEC-linked-triples data stored in AVRO_DIRECTORY and extracts necessary data, such as lemmatized version of the triple (subject, relation, object). The script writes all the data to files in "OPIEC_read" directory.

Prepare_data.py
Script read the data from "OPIEC_read" directory and performs preprocessing and test-train-validation split. As a result, the script saves three datasets in the dictionary format: "Subset_1", "Subset_2" and "Full_data" in pkl file.

Gold_standard_construction.py
The scripts performs filterting of the relations and retrieval of the synonyms from WordNet.
The result is different from "Gold_standard_manual.csv" because it produces temporary file for manual inspection.

Model_results.py
The script introduces functions necessary for model training - "train_save" and evaluation of the results without hyperparameter tuning: "clustering_result", "clustering_with_params", "eval_link"

Model_selection.py
Script performs random search over hyperparameter grid and the function "hp_search_kge", first, trains models for each of the combination of parameters and then, trains and saved model with the best performance on gold standard.

Model_selection_clustering.py
First, the script performs exhaustive search over hyperparameter grif for clustering algorithm. Then, the saved KGE models are restored and the clustering for each of the hyperparameter values is performed and evaluated on gold standard.


