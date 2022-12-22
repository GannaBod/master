import pandas as pd
import random
import numpy as np
from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from clusteval import clusteval
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from Read_corpus import load_dict


def print_evaluation(ranks):
    print('Mean Rank:', mr_score(ranks)) 
    print('Mean Reciprocal Rank:', mrr_score(ranks)) 
    print('Hits@1:', hits_at_n_score(ranks, 1))
    print('Hits@10:', hits_at_n_score(ranks, 10))
    print('Hits@100:', hits_at_n_score(ranks, 100))


def train_save(model, data, model_name):
    model.fit(data['train'],                                      
            early_stopping=False),                          
    save_model(model, 'models/'+model_name)

def evaluate_link(model, data, model_name, entities_subset:bool):
    data, entities, relations= load_dict('Subset_1')
    X_filter = np.concatenate([data['train'], data['valid'], data['test']], 0)
    if entities_subset:
        random.seed(1)
        random.shuffle(entities)
        ent_eval=entities[:2000]
        ranks = evaluate_performance(data['test'], 
                                    model=model, 
                                    filter_triples=X_filter, entities_subset=ent_eval)
    else:
        ranks = evaluate_performance(data['test'], 
                                    model=model, 
                                    filter_triples=X_filter)    
    print(model_name,":")
    print_evaluation(ranks)
    return {'model': model_name, 'mr': mr_score(ranks), 'mrr': mrr_score(ranks), 'hits@1': hits_at_n_score(ranks, 1), 'hits@10': hits_at_n_score(ranks, 10), 'hits@100': hits_at_n_score(ranks, 100)}


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

def Model_results_link():
     # evaluate link prediction performance
    link_pr_result=pd.DataFrame()
    data, entities, relations= load_dict('Subset_1')
    for (modelpath, model_name) in [("models/TransE_bl",'TransE_bl'), ("models/ComplEx_bl",'ComplEx_bl'), ("models/HolE_bl", 'HolE_bl'), ("models/DistMult_bl", 'DistMult_bl'),
    ('models/TransE_best_sb1', 'TransE_best_sb1'), ('models/ComplEx_best_sb1', 'ComplEx_best_sb1'), ('models/HolE_best_sb1', 'HolE_best_sb1'), ('models/DistMult_best_sb1', 'DistMult_best_sb1')]:
        model=restore_model(modelpath)
        result=evaluate_link(model, data, model_name, False)
        link_pr_result=link_pr_result.append(result, ignore_index=True)
    data, entities, relations= load_dict('Full_data')
    model=restore_model('models/TransE_full')
    result=evaluate_link(model, data, 'TransE_full', True)
    link_pr_result=link_pr_result.append(result, ignore_index=True)
    link_pr_result.to_csv('Link_prediction_result.csv')

if __name__ == "__main__":
    Model_results_baseline()
    Model_results_link()

      