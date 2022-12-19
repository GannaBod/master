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
from Models_results import clustering_result, gold_st, train_save

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
            element.update({'optimizer_parameters':{'lr': 1e-3}}) #what if we leave it default small

    return dict_list


def hp_search_kge(model_class, model_name, param_dict_list, data, gs, relations):
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
        optimizer= parameter_combination['optimizer'],
        initializer=parameter_combination['initializer'],
        regularizer_params=parameter_combination['regularizer_parameters'],
        optimizer_params=parameter_combination['optimizer_parameters'],
        verbose= parameter_combination['verbose'])

        model.fit(data['train'],                          
        early_stopping=False) 
        print('Parameter combination:', parameter_combination)
        cl_result=clustering_result(model, model_class, gs_rels, gs_clusters)
        print(cl_result)
        df=df.append({"Model": model_class, 'params': parameter_combination, 'ARS': cl_result['ARS'] }, ignore_index=True)
    df.to_csv('results/Model_selection'+model_name+'.csv')


    #df=pd.read_csv('Model_selection_kge_TransE.csv')
    #retrain the best model

    best_params=df.sort_values(by=['ARS'], ascending=False)['params'].iloc[0]
    model=model_class(batches_count=best_params['batches_count'],
        seed= best_params['seed'],
        epochs= best_params['epochs'],
        k= best_params['k'],
        eta= best_params['eta'],
        loss= best_params['loss'],
        optimizer= best_params['optimizer'],
        initializer=best_params['initializer'],
        regularizer_params=best_params['regularizer_parameters'],
        optimizer_params=best_params['optimizer_parameters'],
        verbose= best_params['verbose'])
    train_save(model, data, model_name) #change model name for saving

def second_best(table_path, data_path, model_class, model_name):
    data, entities, relations= load_dict(data_path)
    df=pd.read_csv(table_path)
    second_best_params=df.sort_values(by=['ARS'], ascending=False)['params'].iloc[1]
    second_best_params=eval(second_best_params)
    model=model_class(batches_count=second_best_params['batches_count'],
        seed= second_best_params['seed'],
        epochs= second_best_params['epochs'],
        k= second_best_params['k'],
        eta= second_best_params['eta'],
        loss= second_best_params['loss'],
        optimizer= second_best_params['optimizer'],
        initializer=second_best_params['initializer'],
        regularizer_params=second_best_params['regularizer_parameters'],
        optimizer_params=second_best_params['optimizer_parameters'],
        verbose= second_best_params['verbose'])
    train_save(model, data, model_name) #change model name for saving


def train_best_params(table_path, data_path, model_class, model_name):
    data, entities, relations= load_dict(data_path)
    df=pd.read_csv(table_path)
    best_params=df.sort_values(by=['ARS'], ascending=False)['params'].iloc[1] #[1]

    #best_params=df.sort_values(by=['ARS'], ascending=False)['params'][0]

    best_params=eval(best_params)
    model=model_class(batches_count=best_params['batches_count'],
        seed= best_params['seed'],
        epochs= best_params['epochs'],
        k= best_params['k'],
        eta= best_params['eta'],
        loss= best_params['loss'],
        optimizer= 'sgd', #best_params['optimizer'], # TODO
        initializer=best_params['initializer'],
        regularizer_params=best_params['regularizer_parameters'],
        optimizer_params=best_params['optimizer_parameters'],
        verbose= best_params['verbose'])
    
    train_save(model, data, model_name) 

def Model_selection():
    data, entities, relations= load_dict('Subset_1')
    param_grid = {
                    "batches_count": [100],
                    "seed": [0],
                    "epochs": [20, 50, 100],
                    "k": [100, 200],
                    "eta": [1, 5, 10],
                    "loss": ["pairwise", 'nll', 'absolute_margin', 'multiclass_nll', 'self_adversarial'],
                    "optimizer": ["adam", 'adagrad','sgd'],
                    "initializer":['normal', 'uniform', 'xavier'],
                    "verbose": [True]
                }
    regul_params ={"p": [1,2,3], "lambda": [1e-3, 1e-5]}
    param_dict_list=sample_param_grid(param_grid, 7, regul_params)

    for (model, model_name) in [(TransE, 'TransE_best_sb1'), (ComplEx, 'ComplEx_best_sb1'), (HolE, 'HolE_best_sb1'), (DistMult, 'DistMult_best_sb1')]:
        hp_search_kge(model, model_name, param_dict_list, data, 'Gold_standard_manual.csv', relations)
    for (table_path, model_class, model_name) in [('results/Model_selectionTransE_best_sb1.csv', TransE, 'TransE_2nd_best_sb1'), ('results/Model_selectionComplEx_best_sb1.csv', ComplEx, 'ComplEx_2nd_best_sb1'), ('results/Model_selectionHolE_best_sb1.csv', HolE, 'HolE_2nd_best_sb1'), ('results/Model_selectionDistMult_best_sb1.csv', DistMult, 'DistMult_2nd_best_sb1')]:
        second_best(table_path, 'Subset_1', model_class, model_name)



if __name__ == "__main__":

    from ampligraph.latent_features import set_entity_threshold
    set_entity_threshold(2200000)

    # data, entities, relations= load_dict('Preprocessed')
    # param_grid = {
    #                 "batches_count": [100],
    #                 "seed": [0],
    #                 "epochs": [20, 50, 100],
    #                 "k": [100, 200],
    #                 "eta": [1, 5, 10],
    #                 "loss": ["pairwise", 'nll', 'absolute_margin', 'multiclass_nll', 'self_adversarial'],
    #                 "optimizer": ["adam", 'adagrad','sgd'],
    #                 "initializer":['normal', 'uniform', 'xavier'],
    #                 "verbose": [True]
    #             }
    # regul_params ={"p": [1,2,3], "lambda": [1e-3, 1e-5]}
    # param_dict_list=sample_param_grid(param_grid, 7, regul_params)
    # early_stopping_params = { 'x_valid': data['valid'],   
    #                       'criteria': 'mrr',    
    #                       'burn_in': 50,      #not working to train models like this... but what about model selection? did it work?
    #                       'check_interval': 50, 
    #                       'stop_interval': 2,   
    #                       'corrupt_side':'s,o'   
    #                     }
    # for (model, model_name) in [(TransE, 'TransE_best_3'), (ComplEx, 'ComplEx_best_3'), (HolE, 'HolE_best_3'), (DistMult, 'DistMult_best_3')]:#(HolE, 'HolE_best_3'), (DistMult, 'DistMult_best_3')]:#], ComplEx, HolE, DistMult]   (TransE, 'TransE_best_3'), (ComplEx, 'ComplEx_best_3'),                     
    #     hp_search_kge(model, model_name, param_dict_list, data, 'Gold_standard_ver3.csv', relations)
    
    #retrain model with the best parameters

    #df=pd.DataFrame(columns=['model', 'mr', 'mrr', 'hits@1', 'hits@10', 'hits@100']) 

    #train best models on bigger data
    #for (model_class, model_name, table_path) in [(TransE, 'TransE_best_9', 'Model_selectionTransE_best_3.csv')]: #[(ComplEx, 'ComplEx_best_9', 'Model_selectionComplEx_best_3.csv'), (HolE, 'HolE_best_9', 'Model_selectionHolE_best_3.csv'), (DistMult, 'DistMult_best_9', 'Model_selectionDistMult_best_3.csv')]:#], ComplEx, HolE, DistMult]   (TransE, 'TransE_best_3', 'Model_selectionTransE_best_3.csv'), (ComplEx, 'ComplEx_best_3'),                     
    #    data_path='Subset_9_docs_new'
    #    train_best_params(table_path, data_path, model_class, model_name)

    data_path='Full_data_preproc'
    train_best_params('Model_selectionTransE_best_3.csv', data_path, TransE, 'TransE_full')
    

##train best model -> one clustering combination. is it the same best clustering combination for all?