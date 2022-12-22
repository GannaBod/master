## To run the code, make sure AVRO_DIRECTORY is set to the path, where data is stored.
from Read_corpus import *
from Prepare_data import *
from Gold_standard_construction import *
from Models_results import *
from ModelSelection import *
from ModelSelection_clustering import *
from Human_evaluation import *


# - 1. Extract triples to OPIEC folder

print("Start reading data")
AVRO_SCHEMA_FILE = "./avroschema/TripleLinked.avsc" 
AVRO_DIRECTORY="data"
full_data=extract_triples_full_data(AVRO_SCHEMA_FILE, AVRO_DIRECTORY)

print("Data read. Data preprocessing...")
# -2. Prepare data
Prepare_data_run() 

print("Gold standard construction")
# 3. Construct Gold Standard
Gold_st_construction()

print("Training of the baseline models and evaluation...")
# 4. Baseline model training and evaluation
Model_results_baseline()

# 5. Model selection - KGE
print("Model selection KGE is running...")
Model_selection() 


# 6. Model selection - Clustering
print("Model selection clustering is running...")
Model_selection_clustering()


# 7. Two best models on subset 2: training & evaluation
print("Retrain model on bigger data and evaluate")
Model_results_subset2() #- not in model results but in other script

# 8. Full data model training and evaluation
print("Retrain model on full data and evaluate")
Model_results_full()

# # 9. Link prediction results
print("Evaluation of link prediction performance")
Model_results_link()

# #10. Human evaluation
print("Human evaluation")
Human_evaluation()