import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
import pandas as pd
import os
import random
import pickle
import numpy as np
import ampligraph

from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE, ConvE, ConvKB, RandomBaseline
from ampligraph.evaluation import select_best_model_ranking
from ampligraph.utils import save_model, restore_model

#import requests

#from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import train_test_split_no_unseen

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score


#import re
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
from ampligraph.utils import create_tensorboard_visualizations

from Models_results import load_data, prepare_data, print_evaluation

def model_selection(model_class, data, param_grid, model_name):

    best_model, best_params, best_mrr_train, ranks_test, mrr_test, experimental_history = \
            select_best_model_ranking(model_class, 
                            data['train'], 
                            data['valid'], 
                            data['test'], 
                            param_grid,
                            max_combinations=5, # performs random search-executes 2 models by randomly choosing params
                            use_filter=True, 
                            verbose=True,
                            early_stopping=True)

    best_dict={'best_model':best_model, 'best_params':best_params, 'best_mrr_train':best_mrr_train, 'ranks_test':ranks_test, 'mrr_test':mrr_test, 'experimental_history':experimental_history}
    X_filter = np.concatenate([data['train'], data['valid'], data['test']], 0)
    ranks = evaluate_performance(data['test'], 
                                model=best_model, 
                                filter_triples=X_filter)
    # display the metrics
    print_evaluation(ranks)
    save_model(best_model, './models/'+model_name)
    return best_model, ranks, best_dict


if __name__ == "__main__":

#TODO move to main file, here we leave only examplary functions

    full_data=load_data('./OPIEC_read')[:500]
    data, entities, relations = prepare_data(full_data, 10,20)

# #save data for future use
#     to_pickle = {
#         "entities": entities,
#         "relations": relations,
#         "train_set": data['train'],
#         "test_set": data['test'],
#         "valid_set": data['valid']
#     }
#     with open("SubsetData", 'wb') as handle:
#         pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    param_grid = {
                    "batches_count": [100],
                    "seed": 0,
                    "epochs": [20, 50],
                    "k": [100, 200],
                    "eta": [1, 5, 10],
                    "loss": ["pairwise", 'nll', 'multiclass_nll', 'self_adversarial'],
                    # We take care of mapping the params to corresponding classes
                    "regularizer": ["LP"],
                    "regularizer_params": {
                        "p": [3],
                        "lambda": [1e-5]
                    },
                    "optimizer": ["adam", 'adagrad'],
                    "optimizer_params":{
                        "lr": [1e-3]
                    },
                    "verbose": True
                }

    early_stopping_params = { 'x_valid': data['valid'],   
                          'criteria': 'mrr',    
                          'burn_in': 150,      
                          'check_interval': 50, 
                          'stop_interval': 2,   
                          'corrupt_side':'s,o'  
                        }

    df=pd.DataFrame(columns=['model', 'mr', 'mrr', 'hits@1', 'hits@10', 'hits@100']) 

    #1 TransE
    #save best params and write them down. Save best model.
    model_1, ranks_1, best_dict_1 = model_selection(TransE, data, param_grid, 'TransE_best')
    result_1={'model': 'TransE', 'mr': mr_score(ranks_1), 'mrr': mrr_score(ranks_1), 'hits@1': hits_at_n_score(ranks_1, 1), 'hits@10': hits_at_n_score(ranks_1, 10), 'hits@100': hits_at_n_score(ranks_1, 100), 'best_params':best_dict_1['best_params']}
    df=df.append(result_1, ignore_index=True)

    #2 ComplEx
    model_2, ranks_2, best_dict_2 = model_selection(ComplEx, data, param_grid, 'ComplEx_best')
    result_2={'model': 'ComplEx', 'mr': mr_score(ranks_2), 'mrr': mrr_score(ranks_2), 'hits@1': hits_at_n_score(ranks_2, 1), 'hits@10': hits_at_n_score(ranks_2, 10), 'hits@100': hits_at_n_score(ranks_2, 100), 'best_params':best_dict_2['best_params']}
    df=df.append(result_2, ignore_index=True)
    #3 HolE
    model_3, ranks_3, best_dict_3 = model_selection(HolE, data, param_grid, 'HolE_best')
    result_3={'model': 'HolE', 'mr': mr_score(ranks_3), 'mrr': mrr_score(ranks_3), 'hits@1': hits_at_n_score(ranks_3, 1), 'hits@10': hits_at_n_score(ranks_3, 10), 'hits@100': hits_at_n_score(ranks_3, 100), 'best_params':best_dict_3['best_params']}
    df=df.append(result_3, ignore_index=True)
    #4 DistMult
    model_4, ranks_4, best_dict_4 = model_selection(DistMult, data, param_grid, 'DistMult_best')
    result_4={'model': 'TransE', 'mr': mr_score(ranks_4), 'mrr': mrr_score(ranks_4), 'hits@1': hits_at_n_score(ranks_4, 1), 'hits@10': hits_at_n_score(ranks_4, 10), 'hits@100': hits_at_n_score(ranks_4, 100), 'best_params':best_dict_4['best_params']}
    df=df.append(result_4, ignore_index=True)
    print(df)
    df.to_csv('results_model_selection.csv')
