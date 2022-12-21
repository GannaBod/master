import tensorflow as tf
import pandas as pd
import os
import random
import pickle
import numpy as np
import ampligraph
from sklearn.model_selection import ParameterSampler, ParameterGrid
from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE, ConvE, ConvKB, RandomBaseline
from ampligraph.evaluation import select_best_model_ranking
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance, mr_score, mrr_score, hits_at_n_score, train_test_split_no_unseen
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from clusteval import clusteval
from Read_corpus import load_dict
from Models_results import clustering_result, gold_st, clustering_results_with_params
from ModelSelection import train_best_params


def exhaustive_param_grid(param_grid, regul_params=None):
    param_list = list(ParameterGrid(param_grid))
    return param_list


def hp_search_clustering(model, relations, cl_param_list, pca:bool, model_name, gs):
    df=pd.DataFrame()
    gs_rels, gs_clusters=gold_st(gs, relations)
    E_gs=[]
    probl_v=[]
    for verb in gs_rels:
        try:
            E_gs.append(model.get_embeddings(np.array(verb), embedding_type='relation'))
        except (RuntimeError, TypeError, NameError, IndexError, ValueError):
            print(verb)
            probl_v.append(verb)
    E_gs=np.array(E_gs)  
    prob_i=[i for i,verb in enumerate(gs_rels) if verb in probl_v]
    for i in sorted(prob_i, reverse=True):
        del gs_clusters[i] 
        del gs_rels[i]
    if pca:
      E_gs = PCA(n_components=2, random_state=1).fit_transform(E_gs)
    for cl_param_combination in cl_param_list:
        try:
          ce = clusteval(cluster=cl_param_combination['cluster'], max_clust= cl_param_combination['max_clust'], evaluate=cl_param_combination['evaluate'], linkage=cl_param_combination['linkage'], metric=cl_param_combination['metric']) 
          c=ce.fit(E_gs)
          clusters=c['labx']
          n_cl_opt=len(set(clusters))
          ars=adjusted_rand_score(gs_clusters, clusters)

          df=df.append({"Model": model_name, 'params': cl_param_combination, 'ARS': ars, 'N_cl_opt': n_cl_opt }, ignore_index=True)
        except Exception:
          pass
    result=df.sort_values(by=["ARS"], ascending=False).head(10)
    result.to_csv('Model_selection_clustering'+model_name+'.csv')
    print(result)
    return df

def Model_selection_clustering():
    data, entities, relations= load_dict('Subset_1')
    cl_params={ 'cluster':['kmeans', 'agglomerative', 'hdbscan'], 'max_clust':[25,26,28,30],
    'evaluate':['silhouette', 'dbindex', 'derivative'],
        'linkage':['ward','single','complete','average','weighted','centroid','median'],
        'metric':['euclidean', 'cosine']}
    cl_params_list=exhaustive_param_grid(cl_params)
    for modelpath, model_name in [('models/TransE_best_sb1', 'TransE_best_sb1'), ('models/ComplEx_best_sb1', 'ComplEx_best_sb1'), ('models/HolE_best_sb1', 'HolE_best_sb1'), ('models/DistMult_best_sb1', 'DistMult_best_sb1'),
                                    ('models/TransE_2nd_best_sb1', 'TransE_2nd_best_sb1'), ('models/ComplEx_2nd_best_sb1', 'ComplEx_2nd_best_sb1'), ('models/HolE_2nd_best_sb1', 'HolE_2nd_best_sb1'), ('models/DistMult_2nd_best_sb1', 'DistMult_2nd_best_sb1')]:
        model=restore_model(modelpath)
        hp_search_clustering(model, relations, cl_params_list, True, model_name, 'Gold_standard_manual.csv')

def Model_results_subset2():
    #train best models on bigger data
    data, entities, relations= load_dict('Subset_2')
    gs_rels, gs_clusters=gold_st('Gold_standard_manual.csv', relations)
    for (model_class, model_name, table_path_kge, table_path_clust) in [(TransE, 'TransE_best_sb2', 'Model_selectionTransE_best_sb1.csv', 'Model_selection_clusteringTransE_2nd_best_sb1.csv'),(ComplEx, 'ComplEx_best_sb1', 'Model_selectionComplEx_best_sb1.csv','Model_selection_clusteringComplEx_best_sb1.csv')]:                 
       data_path='Subset_2'
       train_best_params(table_path_kge, data_path, model_class, model_name)
       clustering_results_with_params(model, model_name, gs_rels, gs_clusters, table_path_clust)

def Model_results_full():
    data, entities, relations= load_dict('Full_data')
    data_path='Full_data'
    gs_rels, gs_clusters=gold_st('Gold_standard_manual.csv', relations)
    train_best_params('Model_selectionTransE_best_3.csv', data_path, TransE, 'TransE_full')
    clustering_results_with_params(model, 'TransE_full', gs_rels, gs_clusters, 'Model_selection_clusteringTransE_2nd_best.csv')



if __name__ == "__main__":

    data, entities, relations= load_dict('Preprocessed')
    cl_params={ 'cluster':['kmeans', 'agglomerative', 'hdbscan'], 'max_clust':[25,26,28,30],
    'evaluate':['silhouette', 'dbindex', 'derivative'],
        'linkage':['ward','single','complete','average','weighted','centroid','median'],
        'metric':['euclidean', 'cosine']}
    cl_params_list=exhaustive_param_grid(cl_params)

    model=restore_model('models/TransE_best_9')
    hp_search_clustering(model, relations, cl_params_list, True, 'TransE_best_9', 'Gold_standard_manual.csv')


