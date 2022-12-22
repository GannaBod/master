# Repository for the master's thesis: "Knowledge graph embeddings for semantic relation clustering"

## Abstract

This master’s thesis investigates the usage of knowledge graph embedding (KGE’s) for semantic relation clustering. I analysed the performance of 4 knowledge graph embedding models: TransE, ComplEx, HolE and DistMult in relation clustering of the open information extraction corpus. In my experiments, I performed a random hyperparameter search for KGE model selection and an exhaustive search of clustering model selection. The results showed poor performance in
both stages of evaluation: on the manually constructed gold standard and human evaluation of 25 random sampled clusters. I discussed the possible reasons for the low efficiency of the suggested pipeline for the given task and suggested the directions for future work.

## Data

OPIEC corpus can be loaded from https://drive.google.com/drive/folders/1c3yMKLF7fGjIKjwvj_sQ0icRA02m_gzR?usp=sharing


## Code execution

The code requires python 3.7.

To run the code as described in the thesis:
1. install dependencies with command 
`pip install -r requirements.txt`

2. download the full data in the folder "/data"
3. run main.py file


Already trained models can be downloaded from https://drive.google.com/drive/folders/1Kws7nfF2xDWs7OavhnBy7XKBC3QhBEUq?usp=sharing


The project with the full results and models - this is how the project folder looks like when the code is finished can be found under:
https://drive.google.com/drive/folders/1KmJzU2y1U3dJ7MkNeaOnquXu71qyghHh?usp=sharing


## Code description

*main.py* - file from which the whole experiment work described in thesis can be replicated.
Once launched, it executes all code with various models and training procedure. 


*Read_corpus.py* -the script reads the OPIEC-linked-triples data stored in AVRO_DIRECTORY and extracts necessary data, such as lemmatized version of the triple (subject, relation, object). The script writes all the data to files in "OPIEC_read" directory.

*Prepare_data.py* - script reads the data from "OPIEC_read" directory and performs preprocessing and test-train-validation split. As a result, the script saves three datasets in the dictionary format: "Subset_1", "Subset_2" and "Full_data" in pkl file.

*Gold_standard_construction.py* -the scripts performs filterting of the relations and retrieval of the synonyms from WordNet.
The result is different from "Gold_standard_manual.csv" because it produces temporary file for manual inspection.

*Model_results.py* -the script introduces functions necessary for model training - "train_save" and evaluation of the results without hyperparameter tuning: "clustering_result", "clustering_with_params", "eval_link"

*Model_selection.py* -the script performs random search over hyperparameter grid and the function "hp_search_kge", first, trains models for each of the combination of parameters and then, trains and saved model with the best performance on gold standard.

*Model_selection_clustering.py* - first, the script performs exhaustive search over hyperparameter grid for clustering algorithm. Then, the saved KGE models are restored and the clustering for each of the hyperparameter values is performed and evaluated on gold standard.

*Human evaluation.py* -while running the script, the user will be asked to evaluate the clusters of the TransE_full model.

*Visualize.py* -the script provides functions for plotting embeddings in 2D together with labeled relations.


© 2022 Ganna Bodnya
