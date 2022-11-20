import tensorflow as tf
tf.compat.v1.disable_eager_execution()
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
from clusteval import clusteval

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score


#import re
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
from ampligraph.utils import create_tensorboard_visualizations

from Prepare_data import load_data, prepare_data#, load_dict
from Read_corpus import load_dict

# def load_data(DATA_DIRECTORY: str):
#     full_data=pd.DataFrame()
#     i=0
#     for filename in os.listdir(DATA_DIRECTORY):
#         i+=1
#         with open(DATA_DIRECTORY+'/'+filename, 'rb') as file:
#             data = pickle.load(file)
#             #why pandas? maybe numpy straight forward?
#             df=pd.DataFrame(data['triples'])
#         full_data=full_data.append(df)
#         if i>3:
#             break
#     return full_data

# def prepare_data(full_data, valid_size, test_size):
#     X=np.array(full_data.values)
#     data = {}
#     data['train'], data['valid'] = train_test_split_no_unseen(X, test_size=valid_size, seed=1, allow_duplication=False) 
#     data['train'], data['test']=train_test_split_no_unseen(data['train'], test_size=test_size, seed=1, allow_duplication=False)

#     entities=np.array(list(set(full_data.values[:, 0]).union(full_data.values[:, 2])))
#     relations=np.array(list(set(full_data.values[:, 1])))

#     print('Unique entities: ', len(entities))
#     print('Unique relations: ', len(relations))

#     print('Train set size: ', data['train'].shape)
#     print('Test set size: ', data['test'].shape)
#     print('Valid set size: ', data['valid'].shape)
#     print(data['train'])

#     return data, entities, relations

def print_evaluation(ranks):
    print('Mean Rank:', mr_score(ranks)) 
    print('Mean Reciprocal Rank:', mrr_score(ranks)) 
    print('Hits@1:', hits_at_n_score(ranks, 1))
    print('Hits@10:', hits_at_n_score(ranks, 10))
    print('Hits@100:', hits_at_n_score(ranks, 100))

#TODO clean code from not my comments

def train_save(model, data, model_name, early_stopping_params):
    model.fit(data['train'],                                      
            early_stopping=True,                          
            early_stopping_params=early_stopping_params) 
    save_model(model, 'models/'+model_name)
    # X_filter = np.concatenate([data['train'], data['valid'], data['test']], 0)
    # ranks = evaluate_performance(data['test'], 
    #                             model=model, 
    #                             filter_triples=X_filter)
    # print_evaluation(ranks)
    # return {'model': model_name, 'mr': mr_score(ranks), 'mrr': mrr_score(ranks), 'hits@1': hits_at_n_score(ranks, 1), 'hits@10': hits_at_n_score(ranks, 10), 'hits@100': hits_at_n_score(ranks, 100)}

#TODO save all models and present all evaluation results in a table
#for model in list[] ...
#print(df)

def gold_st(DOC_PATH, relations):
    gs=pd.read_csv(DOC_PATH)
    gs_rels=[]
    gs_clusters=[]
    for row in gs.itertuples():
        if row.verb in relations:
            gs_rels.append(row.verb)
            gs_clusters.append(row.cluster)
    print("Gold standard relations number:",  len(gs_rels))
    return gs_rels, gs_clusters

#def link_prediction(model):
def clustering_result(model, model_name, gs_rels, gs_clusters):
    E_gs=[]
    probl_v=[]
    for verb in gs_rels:
        try:
            E_gs.append(model.get_embeddings(np.array(verb), embedding_type='relation'))
        except (RuntimeError, TypeError, NameError, IndexError, ValueError):
            probl_v.append(verb)
    E_gs=np.array(E_gs)  
    prob_i=[i for i,verb in enumerate(gs_rels) if verb in probl_v]
    for i in sorted(prob_i, reverse=True):
        del gs_clusters[i] 
        del gs_rels[i]    

    ce = clusteval(cluster='kmeans', evaluate='silhouette') #in 
    c=ce.fit(E_gs)
    score_table=c['score']
    score_max=score_table['score'].max()
    n_cl_opt=score_table[score_table['score']==score_max]['clusters'].values[0]
    silh_best=score_table['score'].max()
    print('Optimal number of clusters =', n_cl_opt)
    print('Best silhouette score =', silh_best)

    ##DOES NOT MAKE SENSE -> IN CLUSTEVAL WITH CLUSTER WITH DIFF ALG
    #kMeans clustering
    kmeans = KMeans(n_clusters=n_cl_opt, random_state=0).fit(E_gs)
    clusters=kmeans.labels_
    print(len(clusters))
    #print results
    ars=adjusted_rand_score(gs_clusters, clusters)
#ARS_GS
    n_cl_gs=len(set(gs_clusters))
    kmeans = KMeans(n_clusters=n_cl_gs, random_state=0).fit(E_gs)
    clusters=kmeans.labels_
    ars_gs=adjusted_rand_score(gs_clusters, clusters)

    print("Adjusted_rand_score KMeans clustering with"+str(n_cl_opt)+"clusters",(ars))
    return {'Model': model_name, 'N_cl_opt': n_cl_opt, 'silh_best': silh_best, 'ARS_opt': ars, 'ARS_gs': ars_gs}


if __name__ == "__main__":

#TODO move to main file, here we leave only examplary functions

    # full_data=load_data('./OPIEC_read')
    # data, entities, relations = prepare_data(full_data, 100,2000)

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
    data, entities, relations= load_dict('Subset_3_docs_new')

    early_stopping_params = { 'x_valid': data['valid'],   
                          'criteria': 'mrr',    
                          'burn_in': 150,      
                          'check_interval': 50, 
                          'stop_interval': 2,   
                          'corrupt_side':'s,o'  
                        }
    df=pd.DataFrame(columns=['model', 'mr', 'mrr', 'hits@1', 'hits@10', 'hits@100']) 
    gs_rels, gs_clusters=gold_st('Gold_standard_ver3.csv', relations)
    #TODO show progress verbose 

    # # 1. Random Model - works
#there is no get_embeddings method in random baseline model.

    # model=RandomBaseline(verbose=True)
    # result=train_evaluate(model, data, 'RandomBaseline_0', early_stopping_params)
    # df=df.append(result, ignore_index=True)
    #save_model(model, "G:\My Drive\Colab Notebooks\Sessions\models 9 docs\0\RandomBaseline_0")

    # 2. TransE
    #model=TransE(verbose=True)
    #result=train_evaluate(model, data, 'TransE_new',early_stopping_params)
    #df=df.append(result, ignore_index=True)
    model=restore_model('models/TransE_new')
    result=clustering_result(model, 'TransE_best', gs_rels, gs_clusters)
    print(result)

    #save_model(model, '/content/drive/MyDrive/Colab Notebooks/Sessions/models 9 docs/0/TransE_0')
    #3. ComplEx         
    # model=ComplEx(verbose=True)
    # #model=restore_model('models/ComplEx_0')
    # result=train_evaluate(model, data, 'ComplEx_new', early_stopping_params)
    #df=df.append(result, ignore_index=True)
    #save_model(model, '/content/drive/MyDrive/Colab Notebooks/Sessions/models 9 docs/0/ComplEx_0')
    # # 3. HolE         
    # model=HolE(verbose=True)
    # result=train_evaluate(model, data, 'HolE_pre', early_stopping_params)
    # df=df.append(result, ignore_index=True)
    # # save_model(model, '/content/drive/MyDrive/Colab Notebooks/Sessions/models 9 docs/0/HolE_0')

    # # 3. DistMult         
    # model=DistMult(verbose=True)
    # result=train_evaluate(model, data, 'DistMult_pre', early_stopping_params)
    # df=df.append(result, ignore_index=True)
    #save_model(model, '/content/drive/MyDrive/Colab Notebooks/Sessions/models 9 docs/0/DistMult_0')


    
    #print(df)
    #df.to_csv('results_Preprocessed2.csv')
