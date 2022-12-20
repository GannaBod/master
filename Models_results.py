import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
import os
import random
import pickle
import numpy as np
import ampligraph

from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE, ConvE, ConvKB, RandomBaseline
from ampligraph.evaluation import select_best_model_ranking
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import train_test_split_no_unseen

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from clusteval import clusteval

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
from ampligraph.utils import create_tensorboard_visualizations

from Prepare_data import load_data, prepare_data#, load_dict
from Read_corpus import load_dict
#from ModelSelection import train_best_params

from sklearn_extra.cluster import KMedoids


def print_evaluation(ranks):
    print('Mean Rank:', mr_score(ranks)) 
    print('Mean Reciprocal Rank:', mrr_score(ranks)) 
    print('Hits@1:', hits_at_n_score(ranks, 1))
    print('Hits@10:', hits_at_n_score(ranks, 10))
    print('Hits@100:', hits_at_n_score(ranks, 100))

#TODO clean code from not my comments

def train_save(model, data, model_name):#, early_stopping_params):
    model.fit(data['train'],                                      
            early_stopping=False),                          
            #early_stopping_params=early_stopping_params) 
    save_model(model, 'models/'+model_name)

def evaluate_link(model, data, model_name, entities_subset: None):
    X_filter = np.concatenate([data['train'], data['valid'], data['test']], 0)
    ranks = evaluate_performance(data['test'], 
                                model=model, 
                                filter_triples=X_filter, entities_subset=entities_subset)
    print(model_name,":")
    print_evaluation(ranks)
    return {'model': model_name, 'mr': mr_score(ranks), 'mrr': mrr_score(ranks), 'hits@1': hits_at_n_score(ranks, 1), 'hits@10': hits_at_n_score(ranks, 10), 'hits@100': hits_at_n_score(ranks, 100)}

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
    print("Verbs not found in training data:", len(prob_i))
    for i in sorted(prob_i, reverse=True):
        del gs_clusters[i] 
        del gs_rels[i]    
    E_gs = PCA(n_components=2, random_state=1).fit_transform(E_gs)
    ce = clusteval(cluster='agglomerative', evaluate='silhouette') #in 
    c=ce.fit(E_gs)
    clusters=c['labx']
    n_cl_opt=len(set(clusters))
    ars=adjusted_rand_score(gs_clusters, clusters)
    print("Adjusted_rand_score Agglomerative  clustering with "+str(n_cl_opt)+" clusters",(ars))
    return {'Model': model_name, 'N_cl_opt': n_cl_opt, 'ARS': ars}

def clustering_results_with_params(model, model_name, gs_rels, gs_clusters, table_path):
    E_gs=[]
    probl_v=[]
    for verb in gs_rels:
        try:
            E_gs.append(model.get_embeddings(np.array(verb), embedding_type='relation'))
        except (RuntimeError, TypeError, NameError, IndexError, ValueError):
            probl_v.append(verb)
    E_gs=np.array(E_gs)  
    prob_i=[i for i,verb in enumerate(gs_rels) if verb in probl_v]
    print("Verbs not found in training data:", len(prob_i))
    for i in sorted(prob_i, reverse=True):
        del gs_clusters[i] 
        del gs_rels[i]    
    E_gs = PCA(n_components=2, random_state=1).fit_transform(E_gs)
    df=pd.read_csv(table_path)
    best_params=df.sort_values(by=['ARS'], ascending=False)['params'].iloc[0] #[1]
    best_params=eval(best_params)
    ce = clusteval(
        cluster=best_params['cluster'], max_clust= best_params['max_clust'], evaluate=best_params['evaluate'], linkage=best_params['linkage'], metric=best_params['metric']) #in 
    c=ce.fit(E_gs)
    clusters=c['labx']
    n_cl_opt=len(set(clusters))
    ars=adjusted_rand_score(gs_clusters, clusters)
    print("Adjusted_rand_score with best parameters: ",(ars))
    print({'Model': model_name, 'N_cl_opt': n_cl_opt, 'ARS': ars})
    return {'Model': model_name, 'N_cl_opt': n_cl_opt, 'ARS': ars, 'clusters': clusters}

def Model_results_baseline():
    data, entities, relations= load_dict('Subset_1')
    for model, model_name in [(TransE(verbose=True), 'TransE_bl'), (ComplEx(verbose=True), 'ComplEx_bl'), (HolE(verbose=True), 'HolE_bl'), (DistMult(verbose=True), 'DistMult_bl')]:
        train_save(model, data, model_name)
    gs_rels, gs_clusters=gold_st('Gold_standard_manual.csv', relations)
    eval_baseline=pd.DataFrame()
    for model_path, model_name in [('models/TransE_bl', 'TransE_bl'), ('models/ComplEx_bl', 'ComplEx_bl'), ('models/HolE_bl', 'HolE_bl'), ('models/DistMult_bl', 'DistMult_bl')]:
        model= restore_model(model_path)
        eval_baseline=eval_baseline.append(clustering_result (model, model_name, gs_rels, gs_clusters), ignore_index=True)
    print(eval_baseline)
    eval_baseline.to_csv("Baseline_results.csv")

# def Model_results_subset2():
#     #train best models on bigger data
#     gs_rels, gs_clusters=gold_st('Gold_standard_manual.csv', relations)
#     for (model_class, model_name, table_path) in [(TransE, 'TransE_best_9', 'Model_selectionTransE_best_3.csv'),(TransE, 'TransE_best_9', 'Model_selectionTransE_best_3.csv')]: #[(ComplEx, 'ComplEx_best_9', 'Model_selectionComplEx_best_3.csv'), (HolE, 'HolE_best_9', 'Model_selectionHolE_best_3.csv'), (DistMult, 'DistMult_best_9', 'Model_selectionDistMult_best_3.csv')]:#], ComplEx, HolE, DistMult]   (TransE, 'TransE_best_3', 'Model_selectionTransE_best_3.csv'), (ComplEx, 'ComplEx_best_3'),                     
#        data_path='Subset_2'
#        train_best_params(table_path, data_path, model_class, model_name)
#        clustering_results_with_params(model, 'TransE_full', gs_rels, gs_clusters, 'Model_selection_clusteringTransE_2nd_best.csv')

# def Model_results_full():
#     data_path='Full_data'
#     gs_rels, gs_clusters=gold_st('Gold_standard_manual.csv', relations)
#     train_best_params('Model_selectionTransE_best_3.csv', data_path, TransE, 'TransE_full')
#     clustering_results_with_params(model, 'TransE_full', gs_rels, gs_clusters, 'Model_selection_clusteringTransE_2nd_best.csv')
    
def Model_results_link():
     # 2. evaluate link prediction
    link_pr_result=pd.DataFrame()
    data, entities, relations= load_dict('Subset_1')
    for (modelpath, model_name) in [("models/TransE_bl",'TransE_bl'), ("models/ComplEx_bl",'ComplEx_bl'), ("models/HolE_bl", 'HolE_bl'), ("models/DistMult_bl", 'DistMult_bl'),
    ('models/TransE_best_sb1', 'TransE_best_sb1'), ('models/ComplEx_best_sb1', 'ComplEx_best_sb1'), ('models/HolE_best_sb1', 'HolE_best_sb1'), ('models/DistMult_best_sb1', 'DistMult_best_sb1')]:
        model=restore_model(modelpath)
        result=evaluate_link(model, data, model_name)
        link_pr_result=link_pr_result.append(result, ignore_index=True)
    data, entities, relations= load_dict('Full_data')
    model=restore_model('models/TransE_full')
    result=evaluate_link(model, data, 'TransE_full')
    link_pr_result=link_pr_result.append(result, ignore_index=True)
    link_pr_result.to_csv('Link_prediction_result.csv')

if __name__ == "__main__":
# 1. train and save, evaluate default with default clustering. 


    #data, entities, relations= load_dict('Subset_3_docs_new')
    data, entities, relations= load_dict('Full_data_preproc')

    #gs_rels, gs_clusters=gold_st('Gold_standard_ver3.csv', relations)

    # #evaluate default models
    # eval_baseline=pd.DataFrame()
    # for model_path, model_name in [('models/TransE_preproc', 'TransE_bl'), ('models/ComplEx_preproc', 'ComplEx_bl'), ('models/HolE_pre', 'HolE_bl'), ('models/DistMult_pre', 'DistMult_bl')]:
    #     model= restore_model(model_path)
    #     eval_baseline=eval_baseline.append(clustering_result (model, model_name, gs_rels, gs_clusters), ignore_index=True)
    # print(eval_baseline)
    # eval_baseline.to_csv("Baseline_results.csv")

    # model=restore_model("models/TransE_full")
    # clustering_results_with_params(model, 'TransE_full', gs_rels, gs_clusters, 'Model_selection_clusteringTransE_2nd_best.csv')


    #
    #model=TransE(verbose=True)
    #train_save(model, data, "TransE_preproc", early_stopping_params)


    #model=restore_model("models/DistMult_pre")
    #result=clustering_result(model, 'DistMult_default', gs_rels, gs_clusters)
    #print(result)

    # 2. evaluate link prediction
    #link_pr_result=pd.DataFrame()
    #transe=restore_model("models/TransE_preproc")
    # complex=restore_model("models/ComplEx_preproc")
    # hole=restore_model("models/HolE_pre")
    # distmult=restore_model("models/DistMult_pre")

    # for (model, model_name) in [(transe,'TransE_d')]: #,(complex,'ComplEx_d'), (hole, 'HolE_d'), (distmult, 'DistMult_d')]:
    #     result=evaluate_link(model, data, model_name)
    #     link_pr_result=link_pr_result.append(result, ignore_index=True)
    #     link_pr_result.to_csv('Link_prediction_result.csv')    

    model=restore_model('models/TransE_full')
    random.seed(1)
    random.shuffle(entities)
    print(len(entities))
    ent_eval=entities[:2000]
    print(ent_eval[:3])
    result=evaluate_link(model, data, 'TransE_full', entities_subset=ent_eval)
    print(result)


      