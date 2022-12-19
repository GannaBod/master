from Read_corpus import *
from Prepare_data import *
from Gold_standard_construction import *
from Models_results import *
from ModelSelection import *
from ModelSelection_clustering import *
from Human_evaluation import *


# - 1. Extract triples to OPIEC folder

AVRO_SCHEMA_FILE = "./avroschema/TripleLinked.avsc" 
AVRO_DIRECTORY="data"
full_data=extract_triples_full_data(AVRO_SCHEMA_FILE, AVRO_DIRECTORY)

# -2. Prepare data
Prepare_data_run(1) #1 #2 or #3 - sb1, sb2, full

# 3. Construct Gold Standard
Gold_st_construction()

# 4. Baseline model training and evaluation
Model_results_baseline()

# 5. Model selection - KGE
Model_selection()  # TODO - check


# 6. Model selection - Clustering
Model_selection_clustering()


# 7. Two best models on subset 2: training & evaluation
Model_results_subset2()

# 8. Full data model training and evaluation
Model_results_full()

# 9. Link prediction results
Model_results_link()

#10. Human evaluation
Human_evaluation()