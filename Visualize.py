import pandas as pd
import numpy as np
from ampligraph.utils import restore_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from Models_results import *
from Read_corpus import load_dict

def visualize(gs_rels, model, gs_clusters, clusters, model_name):
    E_gs=model.get_embeddings(np.array(gs_rels), embedding_type='relation')
    #TODO min function
    embeddings_2d = PCA(n_components=2).fit_transform(E_gs)
    # Create a dataframe to plot the embeddings using scatterplot
    df = pd.DataFrame({"gs_rel": gs_rels, "gs_clusters": gs_clusters, "alg_clusters":clusters,
                        "embedding1": embeddings_2d[:, 0], "embedding2": embeddings_2d[:, 1]})

    fig, axes = plt.subplots(1, 2, figsize=(35, 10), sharex=True, sharey=True)
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
    plt.savefig('Plot'+model_name)

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

    data, entities, relations=load_dict('Subset_1')
    gs_rels, gs_clusters =gold_st("Gold_standard_manual.csv", relations)
    model=restore_model("models/TransE_bl")
    cl=clustering_results_with_params(model, "TransE_bl", gs_rels, gs_clusters, 'results/Model_selection_clusteringTransE_best_sb1.csv')
    clusters=cl['clusters']
    visualize(gs_rels, model, gs_clusters, clusters, "Clusters plotted in 2D")

    