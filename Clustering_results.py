import pandas as pd
import os
import random
import pickle
import numpy as np
import ampligraph

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

from ampligraph.utils import save_model, restore_model
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
from ampligraph.utils import create_tensorboard_visualizations
from clusteval import clusteval

from Models_results import load_data, prepare_data
#put here visualization functions

# with open("SubsetData", 'rb') as file:
#     data=pickle.load(file)

# entities=data["entities"]
# relations=data["relations"]
# data['train']=data["train_set"]
# data['test']=data["test_set"]
# data['valid']=data["valid_set"]
def prepare_gold_st(DOC_PATH, data):
    gs=pd.read_csv(DOC_PATH)
    gs=gs.drop(columns=['triples_id'])
    gs['triples']=gs['triples'].apply(lambda x: eval(x))
    gs_rels=[]
    gs_clusters=[]
    #TODO it's a list with repeated verbs - duplicated clustering data. Does not make sense.
    
    #TODO only train data? why?
    df=pd.DataFrame(data['train'], columns=[['subject','predicate','object']]) 
    rels = np.array(list(set(df.values[:, 1]))) #relations set

    for row in gs.itertuples():
        if row.verb in rels:
            gs_rels.append(row.verb)
            gs_clusters.append(row.cluster)
    return gs, gs_rels, gs_clusters

def print_info(data, entities, relations):
    print('Unique entities: ', len(entities))
    print('Unique relations: ', len(relations))

    print('Train set size: ', data['train'].shape)
    print('Test set size: ', data['test'].shape)
    print('Valid set size: ', data['valid'].shape)

# def random_baseline(gs_rels, model, gs_clusters):
#     #E=np.random(len(gs_rels), 300) #maybe 2 dimensions from the very beginning
#      #embeddings for gold standard
#     E_gs=model.get_embeddings(np.array(gs_rels), embedding_type='relation')
#     # silhouette_avg = silhouette_score(E_gs, clusters)
#     # print("For n_clusters = 5, the average silhouette_score on GS is :", silhouette_avg)

#     ce = clusteval(evaluate='silhouette')
#     c=ce.fit(E_gs)
#     score_table=c['score']
#     score_max=score_table['score'].max()
#     n_cl_opt=score_table[score_table['score']==score_max]['clusters'].values[0]
#     silh_best=score_table['score'].max()
#     print('Optimal number of clusters =', n_cl_opt)
#     print('Best silhouette score =', silh_best)

#     #kMeans clustering
#     kmeans = KMeans(n_clusters=n_cl_opt, random_state=0).fit(E_gs)
#     clusters=kmeans.labels_
#     #print results
#     ars=adjusted_rand_score(gs_clusters, clusters)
#     print("Adjusted_rand_score KMeans clustering with"+str(n_cl_opt)+"clusters",(ars))

def cluster(gs_rels, model, gs_clusters):
    #embeddings for gold standard
    E_gs=model.get_embeddings(np.array(gs_rels), embedding_type='relation')
    # silhouette_avg = silhouette_score(E_gs, clusters)
    # print("For n_clusters = 5, the average silhouette_score on GS is :", silhouette_avg)

    ce = clusteval(evaluate='silhouette')
    c=ce.fit(E_gs)
    score_table=c['score']
    score_max=score_table['score'].max()
    n_cl_opt=score_table[score_table['score']==score_max]['clusters'].values[0]
    silh_best=score_table['score'].max()
    print('Optimal number of clusters =', n_cl_opt)
    print('Best silhouette score =', silh_best)

    #kMeans clustering
    kmeans = KMeans(n_clusters=n_cl_opt, random_state=0).fit(E_gs)
    clusters=kmeans.labels_
    #print results
    ars=adjusted_rand_score(gs_clusters, clusters)
    print("Adjusted_rand_score KMeans clustering with"+str(n_cl_opt)+"clusters",(ars))
    return ars, silh_best, n_cl_opt, clusters

def visualize(gs_rels, model, gs_clusters, clusters):
    E_gs=model.get_embeddings(np.array(gs_rels), embedding_type='relation')
    #TODO min function
    embeddings_2d = PCA(n_components=2).fit_transform(E_gs)
    # Create a dataframe to plot the embeddings using scatterplot
    df = pd.DataFrame({"gs_rel": gs_rels, "gs_clusters": gs_clusters, "alg_clusters":clusters,
                        "embedding1": embeddings_2d[:, 0], "embedding2": embeddings_2d[:, 1]})

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    axes = axes.ravel()
    x=df["embedding1"]
    y=df["embedding2"]
    sns.scatterplot(data=df, x="embedding1", y="embedding2", hue="gs_clusters", ax=axes[0])
    axes[0].set_title('Clusters acc. to GS')
    texts = []
    for i in range(len(x)):
        t = axes[0].text(x[i], y[i], df['gs_rel'][i], ha='center', va='center')
        texts.append(t)
    adjust_text(texts, ax=axes[0])

    sns.scatterplot(data=df, x="embedding1", y="embedding2", hue="alg_clusters", ax=axes[1])
    axes[1].set_title('Clusters acc. to algorithm')
    texts = []
    for i in range(len(x)):
        t = axes[1].text(x[i], y[i], df['gs_rel'][i], ha='center', va='center')
        texts.append(t)
    adjust_text(texts, ax=axes[1])
    plt.show()

def save_results(model_name, ars, silh_best, n_cl_opt):
    results=pd.read_csv('results.csv')
    if 'ars' not in results.columns:
        print('not in results')  
        results['ars']=np.nan         
    results['ars'][results['model']==model_name]=ars
    if 'Silh_best' not in results.columns:
        results['Silh_best']=np.nan
    results['Silh_best'][results['model']==model_name]=silh_best
    if 'Nbr_cl_opt' not in results.columns:
        results['Nbr_cl_opt']=np.nan
    results['Nbr_cl_opt'][results['model']==model_name]=n_cl_opt
    results.to_csv('results.csv', index=False)

if __name__ == "__main__":

    #Load data
    full_data=load_data('./OPIEC_read')
    data, entities, relations = prepare_data(full_data, 100,2000)
    #entity_id_map = {ent_name: id for id, ent_name in enumerate(entities)}
    #relation_id_map = {rel: id for id, rel in enumerate(relations)}

    gs, gs_rels, gs_clusters =prepare_gold_st("Gold_standard_ver2.csv", data)
    print_info(data, entities, relations)

    #for model in [model_list]
    #cluster&viz & save results
    model=restore_model('./models/TransE_1')

    #error cause only 1 latent feature -> 
    model_rb=restore_model('./models/RandomBaseline') #remove None from results record

    #1 visualization of gold standard verbs, golden cluster labels.
    ars, silh_best, n_cl_opt, clusters = cluster(gs_rels, model, gs_clusters)
    visualize(gs_rels, model, gs_clusters, clusters)
    save_results('TransE_1', ars, silh_best, n_cl_opt) 

    #TODO what about diff. models and correlation - random baseline does not work for visualization
    #can I think of my own random baseline ? -> embeddings of size like transE but not trained at all?
    ars2, silh_best2, n_cl_opt2, clusters = cluster(gs_rels, model_rb, gs_clusters)

    
    # ars_2, silh_best_2, n_cl_opt_2 = cluster_and_visualize(gs_rels, model_rb, gs_clusters)
    save_results('RandomBaseline', ars2, silh_best2, n_cl_opt2) 




   