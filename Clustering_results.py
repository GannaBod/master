#RENAME INTO CLUSTER VISUALISATION
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
from Read_corpus import load_dict
#put here visualization functions

# with open("SubsetData", 'rb') as file:
#     data=pickle.load(file)

# entities=data["entities"]
# relations=data["relations"]
# data['train']=data["train_set"]
# data['test']=data["test_set"]
# data['valid']=data["valid_set"]
# def prepare_gold_st(DOC_PATH, data): #long function to drop unnecessary columns
#     gs=pd.read_csv(DOC_PATH)
#     gs=gs.drop(columns=['triples_id'])
#     gs['triples']=gs['triples'].apply(lambda x: eval(x))
#     gs_rels=[]
#     gs_clusters=[]
#     #TODO it's a list with repeated verbs - duplicated clustering data. Does not make sense.
    
#     #TODO only train data? why?
#     df=pd.DataFrame(data['train'], columns=[['subject','predicate','object']]) 
#     rels = np.array(list(set(df.values[:, 1]))) #relations set

#     for row in gs.itertuples():
#         if row.verb in rels:
#             gs_rels.append(row.verb)
#             gs_clusters.append(row.cluster)
#     return gs, gs_rels, gs_clusters

#remove from here 
def gold_st(DOC_PATH, relations):
    gs=pd.read_csv(DOC_PATH)
    gs_rels=[]
    gs_clusters=[]
    for row in gs.itertuples():
        if row.verb in relations:
            gs_rels.append(row.verb)
            gs_clusters.append(row.cluster)
    print("Gold standard relations number:",  len(gs_rels))
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

def cluster_eval(gs_rels, model, gs_clusters):
    #embeddings for gold standard
    E_gs=[]
    #gs_clusters=[]
    print("length gs_clusters", len(gs_clusters))
    print("length gs_rels", len(gs_rels))
    probl_v=[]
    for verb in gs_rels:
        try:
            E_gs.append(model.get_embeddings(np.array(verb), embedding_type='relation'))
            #gs_clusters.append()
        except (RuntimeError, TypeError, NameError, IndexError, ValueError):
            print(verb)
            probl_v.append(verb)
            
    E_gs=np.array(E_gs)  
    print('Prob_v:',len(probl_v))
    prob_i=[i for i,verb in enumerate(gs_rels) if verb in probl_v]
    for i in sorted(prob_i, reverse=True):
        del gs_clusters[i] 
        del gs_rels[i]

    #gs_clusters.remove(gs_clusters[gs_rels.index(verb)])
    #gs_rels.remove(verb)      
    print("E_gs:", type(E_gs))
    print("E_gs shape:", E_gs.shape)
    print("length gs_clusters", len(gs_clusters))
    #E_gs=model.get_embeddings(np.array(gs_rels), embedding_type='relation')
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
    print(len(clusters))
    #print results
    ars=adjusted_rand_score(gs_clusters, clusters)
    print("Adjusted_rand_score KMeans clustering with"+str(n_cl_opt)+"clusters",(ars))
    return ars, silh_best, n_cl_opt, clusters

def visualize(gs_rels, model, gs_clusters, clusters, model_name):
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
    #plt.show()
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/Sessions/models 9 docs/0/'+model_name)

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

#READ OR LOAD?
    #Load data #read or load???
    # full_data=load_data('./OPIEC_read')
    # data, entities, relations = prepare_data(full_data, 100,2000)
    # with open("SubsetData", 'rb') as file:
    #     data=pickle.load(file)
    # data['train']=data['train_set']
    # data['test']=data['test_set']
    # data['valid']=data['valid_set']
    # entities=data['entities']
    # relations=data['relations']
    data, entities, relations=load_dict('Subset_3_docs')

    #entity_id_map = {ent_name: id for id, ent_name in enumerate(entities)}
    #relation_id_map = {rel: id for id, rel in enumerate(relations)}

    #gs, gs_rels, gs_clusters =prepare_gold_st("Gold_standard_ver2.csv", data)
    gs, gs_rels, gs_clusters=gold_st('rel_syns.csv', relations)
    print_info(data, entities, relations)

    #for model in [model_list]
    #cluster&viz & save results
    model_1=restore_model("G:\My Drive\Colab Notebooks\Sessions\models 9 docs\HolE_1")


    #error cause only 1 latent feature -> 
    #model_2=restore_model('./models/RandomBaseline_0') #remove None from results record

    #1 visualization of gold standard verbs, golden cluster labels.
    ars, silh_best, n_cl_opt, clusters = cluster_eval(gs_rels, model_1, gs_clusters)
    visualize(gs_rels, model_1, gs_clusters, clusters, "TransE_0")
    save_results('TransE_0', ars, silh_best, n_cl_opt) 

    # #TODO what about diff. models and correlation - random baseline does not work for visualization
    # #can I think of my own random baseline ? -> embeddings of size like transE but not trained at all?
    # ars2, silh_best2, n_cl_opt2, clusters = cluster(gs_rels, model_2, gs_clusters)

    

    # ##3
    # model_3=restore_model('./models/ComplEx_0')

    # ars3, silh_best3, n_cl_opt3, clusters3 = cluster_eval(gs_rels, model_3, gs_clusters)
    # visualize(gs_rels, model_3, gs_clusters, clusters, 'ComplEx0')
    # save_results('ComplEx_0', ars3, silh_best3, n_cl_opt3) 
    # ##4
    # model_4=restore_model('./models/HolE_0')
    # ars4, silh_best4, n_cl_opt4, clusters4 = cluster_eval(gs_rels, model_4, gs_clusters)
    # visualize(gs_rels, model_4, gs_clusters, clusters, 'HolE0')
    # save_results('HolE_0', ars4, silh_best4, n_cl_opt4) 
    # ##5
    # model_5=restore_model('./models/DistMult_0')
    # ars5, silh_best5, n_cl_opt5, clusters5 = cluster_eval(gs_rels, model_5, gs_clusters)
    # visualize(gs_rels, model_5, gs_clusters, clusters, 'DistMult0')
    # save_results('DistMult_0', ars5, silh_best5, n_cl_opt5) 




   