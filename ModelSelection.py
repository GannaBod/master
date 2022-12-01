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
from sklearn.model_selection import ParameterSampler, ParameterGrid
from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE, ConvE, ConvKB, RandomBaseline
from ampligraph.evaluation import select_best_model_ranking
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance, mr_score, mrr_score, hits_at_n_score, train_test_split_no_unseen
#from sklearn.cluster import AgglomerativeClustering, KMeans

#from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score


#import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
#from ampligraph.discovery import find_clusters
#from ampligraph.utils import create_tensorboard_visualizations
from clusteval import clusteval

#from Models_results import load_data, prepare_data, print_evaluation
from Read_corpus import load_dict
from Models_results import clustering_result, gold_st

def sample_param_grid(param_grid, n_iter, regul_params=None):
    rng = np.random.RandomState(1)
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter,  random_state=rng))
    dict_list = [dict((k, v) for (k, v) in d.items())
                    for d in param_list]
    if regul_params is not None:
        reg_params_list=list(ParameterSampler(regul_params, n_iter=n_iter,  random_state=rng))
        reg_dict_list = [dict((k, v) for (k, v) in d.items())
                        for d in reg_params_list]
        if len(reg_dict_list)<len(dict_list):
            reg_dict_list.extend(random.choices(reg_dict_list, k=(len(dict_list)-len(reg_dict_list))))
        
        for i, element in enumerate(dict_list):
            element.update({'regularizer_parameters': reg_dict_list[i]})
            element.update({'optimizer_parameters':{'lr': 1e-3}})

    return dict_list

def exhaustive_param_grid(param_grid, regul_params=None):
    param_list = list(ParameterGrid(param_grid))
    return param_list
    
#Leave for further research section

# def model_selection(model_class, data, param_grid, model_name):

#     best_model, best_params, best_mrr_train, ranks_test, mrr_test, experimental_history = \
#             select_best_model_ranking(model_class, 
#                             data['train'], 
#                             data['valid'], 
#                             data['test'], 
#                             param_grid,
#                             max_combinations=5, # performs random search-executes 2 models by randomly choosing params
#                             use_filter=True, 
#                             verbose=True,
#                             early_stopping=True)

#     best_dict={'best_model':best_model, 'best_params':best_params, 'best_mrr_train':best_mrr_train, 'ranks_test':ranks_test, 'mrr_test':mrr_test, 'experimental_history':experimental_history}
#     save_model(best_model, './models/'+model_name)
#     X_filter = np.concatenate([data['train'], data['valid'], data['test']], 0)
#     ranks = evaluate_performance(data['test'], 
#                                 model=best_model, 
#                                 filter_triples=X_filter)
#     # display the metrics
#     #print_evaluation(ranks)
    
#     return best_model, ranks, best_dict

#params=['silhouette']
#approach 1: first optimize clustering part, then KGE..
#approach 2: first KGE, then clustering
def hp_search_kge(model_class, param_dict_list, data, early_stopping_params, gs, relations):
    gs_rels, gs_clusters=gold_st(gs, relations)
    df=pd.DataFrame()
    for i, parameter_combination in enumerate(param_dict_list):
        print(parameter_combination)

        model=model_class(batches_count=parameter_combination['batches_count'],
        seed= parameter_combination['seed'],
        epochs= parameter_combination['epochs'],
        k= parameter_combination['k'],
        eta= parameter_combination['eta'],
        loss= parameter_combination['loss'],
        #regularizer= parameter_combination['regularizer'],
        optimizer= parameter_combination['optimizer'],
        initializer=parameter_combination['initializer'],
        regularizer_params=parameter_combination['regularizer_parameters'],
        optimizer_params=parameter_combination['optimizer_parameters'],
        verbose= parameter_combination['verbose'])

        model.fit(data['train'],                          
        early_stopping_params=early_stopping_params) 
        print('Parameter combination:', parameter_combination)
        cl_result=clustering_result(model, i, gs_rels, gs_clusters)
        #maybe here exclude silhouette optimzation and leave only kmeans with gs cluster number
        df=df.append({"Model": model_class, 'params': parameter_combination, 'ARS_gs': cl_result['ARS_gs'], 'N_cl_opt': cl_result['N_cl_opt'], 'Silh_best': cl_result['silh_best'], 'ARS_opt': cl_result['ARS_opt'] }, ignore_index=True)
    df.to_csv('Model_selection_kge_DistMult.csv')

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
      E_gs = PCA(n_components=2).fit_transform(E_gs)
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
    result.to_csv('Model_selection_clustering.csv')
    print(result)
    return df



if __name__ == "__main__":

    data, entities, relations= load_dict('Preprocessed')
    ###
    # X =np.concatenate([data['train'], data['valid'], data['test']], 0)[:10000]
    # data = {}
    # data['train'], data['valid'] = train_test_split_no_unseen(X, test_size=10, seed=1, allow_duplication=False) 
    # data['train'], data['test']=train_test_split_no_unseen(data['train'], test_size=10, seed=1, allow_duplication=False)
    # entities=np.array(list(set(X[:, 0]).union(X[:, 2])))
    # relations=np.array(list(set(X[:, 1])))
    ###
    
    param_grid = {
                    "batches_count": [100],
                    "seed": [0],
                    "epochs": [20, 50, 100],
                    "k": [100, 200],
                    "eta": [1, 5, 10],
                    "loss": ["pairwise", 'nll', 'absolute_margin', 'multiclass_nll', 'self_adversarial'],
                    #"regularizer": ["LP"],
                    "optimizer": ["adam", 'adagrad','sgd'],
                    "initializer":['normal', 'uniform', 'xavier'],
                    "verbose": [True]
                }
    regul_params ={"p": [3,4], "lambda": [1e-3, 1e-5]}

    param_dict_list=sample_param_grid(param_grid, 7, regul_params)

    early_stopping_params = { 'x_valid': data['valid'],   
                          'criteria': 'mrr',    
                          'burn_in': 150,      
                          'check_interval': 50, 
                          'stop_interval': 2,   
                          'corrupt_side':'s,o'  
                        }

    cl_params={ 'cluster':['kmeans', 'agglomerative', 'hdbscan'], 'max_clust':[25,26,28,30],
    'evaluate':['silhouette', 'dbindex', 'derivative'],
        'linkage':['ward','single','complete','average','weighted','centroid','median'],
        'metric':['euclidean', 'cosine']}
    cl_params_list=exhaustive_param_grid(cl_params)

    #hp_search_kge(DistMult, param_dict_list, data, early_stopping_params, 'Gold_standard_ver3.csv', relations)
    model=restore_model('models/TransE_best_3')
    hp_search_clustering(model, relations, cl_params_list, True, 'TransE_best', 'Gold_standard_ver3.csv')

    #df=pd.DataFrame(columns=['model', 'mr', 'mrr', 'hits@1', 'hits@10', 'hits@100']) 

    #1 TransE
    #save best params and write them down. Save best model.
    # model_1, ranks_1, best_dict_1 = model_selection(TransE, data, param_grid, 'TransE_best')
    # result_1={'model': 'TransE', 'mr': mr_score(ranks_1), 'mrr': mrr_score(ranks_1), 'hits@1': hits_at_n_score(ranks_1, 1), 'hits@10': hits_at_n_score(ranks_1, 10), 'hits@100': hits_at_n_score(ranks_1, 100), 'best_params':best_dict_1['best_params']}
    # df=df.append(result_1, ignore_index=True)

    # # #2 ComplEx
    # # model_2, ranks_2, best_dict_2 = model_selection(ComplEx, data, param_grid, 'ComplEx_best')
    # # result_2={'model': 'ComplEx', 'mr': mr_score(ranks_2), 'mrr': mrr_score(ranks_2), 'hits@1': hits_at_n_score(ranks_2, 1), 'hits@10': hits_at_n_score(ranks_2, 10), 'hits@100': hits_at_n_score(ranks_2, 100), 'best_params':best_dict_2['best_params']}
    # # df=df.append(result_2, ignore_index=True)
    # # #3 HolE
    # # model_3, ranks_3, best_dict_3 = model_selection(HolE, data, param_grid, 'HolE_best')
    # # result_3={'model': 'HolE', 'mr': mr_score(ranks_3), 'mrr': mrr_score(ranks_3), 'hits@1': hits_at_n_score(ranks_3, 1), 'hits@10': hits_at_n_score(ranks_3, 10), 'hits@100': hits_at_n_score(ranks_3, 100), 'best_params':best_dict_3['best_params']}
    # # df=df.append(result_3, ignore_index=True)
    # # #4 DistMult
    # # model_4, ranks_4, best_dict_4 = model_selection(DistMult, data, param_grid, 'DistMult_best')
    # # result_4={'model': 'TransE', 'mr': mr_score(ranks_4), 'mrr': mrr_score(ranks_4), 'hits@1': hits_at_n_score(ranks_4, 1), 'hits@10': hits_at_n_score(ranks_4, 10), 'hits@100': hits_at_n_score(ranks_4, 100), 'best_params':best_dict_4['best_params']}
    # # df=df.append(result_4, ignore_index=True)
    # print(df)
    # df.to_csv('results_model_selection3docs.csv')
