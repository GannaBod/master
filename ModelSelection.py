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
from Models_results import clustering_result, gold_st, train_save
from ampligraph.latent_features import set_entity_threshold

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
    best_params=df.sort_values(by=['ARS'], ascending=False)['params'].iloc[1] 
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
    #set_entity_threshold(2200000)

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