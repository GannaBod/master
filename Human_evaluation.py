#load clusters.labels_ list from pickle file
import pandas as pd
import numpy as np
import random
from Read_corpus import load_pkl, load_dict
from sklearn.decomposition import PCA
from clusteval import clusteval
from ampligraph.utils import save_model, restore_model
from Prepare_data import save_pkl, load_pkl
from Gold_standard_construction import get_1w_rel 


def get_save_clusters(model, relations, table_path):
    E_gs=[]
    probl_v=[]
    verbs=[]
    for verb in relations:
        try:
            E_gs.append(model.get_embeddings(np.array(verb), embedding_type='relation'))
            verbs.append(verb)
        except (RuntimeError, TypeError, NameError, IndexError, ValueError):
            probl_v.append(verb)
    E_gs=np.array(E_gs)  
    # prob_i=[i for i,verb in enumerate(relations) if verb in probl_v]
    print("Verbs not found in training data:", (len(relations)-len(verbs)))
    # for i in sorted(prob_i, reverse=True):
    #     relations=np.delete(relations, i)
   
    E_gs = PCA(n_components=2, random_state=1).fit_transform(E_gs)
    save_pkl('cluster_data_1w', E_gs)
    df=pd.read_csv(table_path)
    best_params=df.sort_values(by=['ARS'], ascending=False)['params'].iloc[0] #[1]
    best_params=eval(best_params)
    ce = clusteval(
        cluster=best_params['cluster'], min_clust= 2000, max_clust=2001, evaluate=best_params['evaluate'], linkage=best_params['linkage'], metric=best_params['metric']) #in 
    c=ce.fit(E_gs)
    print(c['score'])
    clusters=c['labx']
    rel_clust={'relations':verbs, 'clusters': clusters}
    save_pkl('rel_clusters_1w', rel_clust)
    return clusters


def evaluate_clusters(clusters, relations, ):
    random.seed(1)
    d=pd.DataFrame({'relation':relations,'cluster':clusters})
    d=d.groupby(by='cluster').agg({'relation': lambda x: ", ".join(x)})
    d['list']=d['relation'].apply(lambda x: x.split(','))
    d['length']=d['list'].apply(lambda x: len(x))
    d=d.reset_index()
    print(d['length'].value_counts())

    results=pd.DataFrame(columns=['Cluster', 'Correct'])
    correct=[]
    n_cl=len(set(d['cluster'][d['length']>1]))
    print("The number of clusters larger 1 overall:", n_cl)
    n_cl=input("Number of clusters to estimate:")
    n_cl=int(n_cl)
    random.seed(1)
    to_eval=random.sample(d['cluster'][d['length']>1].tolist(), n_cl)
    if n_cl!=0:
        for cluster in to_eval:
            print("cluster:", cluster)
            rel_list=d['list'][d['cluster']==cluster].tolist()
            rel_list=rel_list[0]
            print("Length of the cluster:", len(rel_list))
            #print("Number of pairs to evaluate:",)
            res = [(a, b) for idx, a in enumerate(rel_list) for b in rel_list[idx + 1:]]
            print("Number of pairs:", len(res))
            n_pairs=len(res)
            n_pairs=input("Number of pairs to evaluate manually:")
            n_pairs=int(n_pairs)
            if n_pairs==0:
                continue
            p_to_eval=random.sample(res, n_pairs)
            evaluation=[]
            for pair in p_to_eval:
                print(pair)
                x=input("Correct?")
                evaluation.append(x)
                print(evaluation)
            rel_freq=evaluation.count('y') / len(evaluation)
            correct.append(rel_freq)
            results=results.append({'Cluster':cluster, 'Correct': rel_freq}, ignore_index=True)
        results=results.append({'Cluster': 'All', 'Correct': sum(correct)/len(correct)}, ignore_index=True)
        results.to_csv('Human evaluation_1w.csv')

if __name__ == "__main__":

    # data, entities, relations= load_dict('Full_data_preproc') #Full_data_preproc')
    # model=restore_model('models/TransE_full')

    # #subsample relations
    # np.random.seed(1)
    # relations_sub=relations[np.random.choice(relations.shape[0], 4000, replace=False)]
    # print(relations_sub[:5])

    # #get and save clusters

    # clusters=get_save_clusters(model, relations_sub, 'Model_selection_clusteringTransE_2nd_best.csv')
    
    # rel_clust=load_pkl('rel_clusters')
    # print(len(rel_clust['clusters']))
    # print(len(rel_clust['relations']))
    # evaluate_clusters(rel_clust['clusters'], rel_clust['relations'])


##Part 2 one-word relations
    #data, entities, relations= load_dict('Full_data_preproc') #Full_data_preproc')
    #model=restore_model('models/TransE_full')

    #subsample relations
    #np.random.seed(1)
    #relations=np.array(get_1w_rel(relations))
    #print(len(relations))
    #relations_sub=relations[np.random.choice(relations.shape[0], 3400, replace=False)]
    #clusters=get_save_clusters(model, relations_sub, 'Model_selection_clusteringTransE_2nd_best.csv')
    
    rel_clust=load_pkl('rel_clusters_1w')
    print(len(rel_clust['clusters']))
    print(len(rel_clust['relations']))
    evaluate_clusters(rel_clust['clusters'], rel_clust['relations'])
