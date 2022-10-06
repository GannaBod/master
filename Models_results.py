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

def load_data(DATA_DIRECTORY: str):
    full_data=pd.DataFrame()
    i=0
    for filename in os.listdir(DATA_DIRECTORY):
        i+=1
        with open(DATA_DIRECTORY+'/'+filename, 'rb') as file:
            data = pickle.load(file)
            #why pandas? maybe numpy straight forward?
            df=pd.DataFrame(data['triples'])
        full_data=full_data.append(df)
        if i>3:
            break
    return full_data

def prepare_data(full_data, valid_size, test_size):
    X=np.array(full_data.values)
    data = {}
    data['train'], data['valid'] = train_test_split_no_unseen(X, test_size=valid_size, seed=0, allow_duplication=False) 
    data['train'], data['test']=train_test_split_no_unseen(data['train'], test_size=test_size, seed=0, allow_duplication=False)

    entities=np.array(list(set(full_data.values[:, 0]).union(full_data.values[:, 2])))
    relations=np.array(list(set(full_data.values[:, 1])))

    print('Unique entities: ', len(entities))
    print('Unique relations: ', len(relations))

    print('Train set size: ', data['train'].shape)
    print('Test set size: ', data['test'].shape)
    print('Valid set size: ', data['valid'].shape)

    return data, entities, relations

def print_evaluation(ranks):
    print('Mean Rank:', mr_score(ranks)) 
    print('Mean Reciprocal Rank:', mrr_score(ranks)) 
    print('Hits@1:', hits_at_n_score(ranks, 1))
    print('Hits@10:', hits_at_n_score(ranks, 10))
    print('Hits@100:', hits_at_n_score(ranks, 100))

#TODO clean code from not my comments

def train_evaluate(model, data, model_name):
    model.fit(data['train'],                                      
            early_stopping=True,                          
            early_stopping_params=early_stopping_params) 
    X_filter = np.concatenate([data['train'], data['valid'], data['test']], 0)
    ranks = evaluate_performance(data['test'], 
                                model=model, 
                                filter_triples=X_filter)
    print_evaluation(ranks)
    save_model(model, 'models/'+model_name)
    return {'model': model, 'mr': mr_score(ranks), 'mrr': mrr_score(ranks), 'hits@1': hits_at_n_score(ranks, 1), 'hits@10': hits_at_n_score(ranks, 10), 'hits@100': hits_at_n_score(ranks, 100)}

#TODO save all models and present all evaluation results in a table
#for model in list[] ...
#print(df)

if __name__ == "__main__":

#TODO move to main file, here we leave only examplary functions

    full_data=load_data('./OPIEC_read')
    data, entities, relations = prepare_data(full_data, 100,2000)

#save data for future use
    to_pickle = {
        "entities": entities,
        "relations": relations,
        "train_set": data['train'],
        "test_set": data['test'],
        "valid_set": data['valid']
    }
    with open("SubsetData", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    early_stopping_params = { 'x_valid': data['valid'],   
                          'criteria': 'mrr',    
                          'burn_in': 150,      
                          'check_interval': 50, 
                          'stop_interval': 2,   
                          'corrupt_side':'s,o'  
                        }
#for model in ['RandomBaseline', 'ComplEx', 'TransE']
    model= RandomBaseline()
    result=train_evaluate(model, data, 'Random_1')
    #restore 
    #evaluate
    df=pd.DataFrame(columns=['model', 'mr', 'mrr', 'hits@1', 'hits@10', 'hits@100'])
    df=df.append(result, ignore_index=True)
    print(df)
