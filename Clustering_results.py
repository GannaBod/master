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

full_data=load_data('./OPIEC_read')
data, entities, relations = prepare_data(full_data, 100,2000)

# with open("SubsetData", 'rb') as file:
#     data=pickle.load(file)

# entities=data["entities"]
# relations=data["relations"]
# data['train']=data["train_set"]
# data['test']=data["test_set"]
# data['valid']=data["valid_set"]

entity_id_map = {ent_name: id for id, ent_name in enumerate(entities)}
relation_id_map = {rel: id for id, rel in enumerate(relations)}

gs=pd.read_csv("Gold_standard_ver2.csv")
gs=gs.drop(columns=['triples_id'])
gs['triples']=gs['triples'].apply(lambda x: eval(x))


print('Unique entities: ', len(entities))
print('Unique relations: ', len(relations))

print('Train set size: ', data['train'].shape)
print('Test set size: ', data['test'].shape)
print('Valid set size: ', data['valid'].shape)


model=restore_model('./models/TransE_1')
#only train data, why?
df=pd.DataFrame(data['train'], columns=[['subject','predicate','object']]) 
rels = np.array(list(set(df.values[:, 1]))) #relations set


##Gold standard clstering quality

#1 visualization of gold standard verbs, golden cluster labels.

gs_rels=[]
gs_clusters=[]
#TODO it's a list with repeated verbs - duplicated clustering data. Does not make sense.
for row in gs.itertuples():
  if row.verb in rels:
    gs_rels.append(row.verb)
    gs_clusters.append(row.cluster)

E_gs=model.get_embeddings(np.array(gs_rels), embedding_type='relation')

#TODO add hierarchical clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(E_gs)
clusters=kmeans.labels_

ars=adjusted_rand_score(gs_clusters, clusters)
print("Adjusted_rand_score",(ars))
silhouette_avg = silhouette_score(E_gs, clusters)
print("For n_clusters = 5, the average silhouette_score on GS is :", silhouette_avg)

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

results=pd.read_csv('results.csv')
results['ars']=[None, ars]
print(results)

##CLUSTEVAL CHOOSE NUMBER OF CLUSTERS WITH AGGLOMERATIVE CLUSTERING AND SILHOUETTE SCORE
ce = clusteval(evaluate='silhouette')
c=ce.fit(E_gs)
print(c.keys())
score_table=c['score']
print(score_table)
score_max=score_table['score'].max()
n_cl_opt=score_table[score_table['score']==score_max]['clusters'][0]
silh_best=score_table['score'].max()
print('Optimal number of clusters =', n_cl_opt)
print('Best silhouette score =', silh_best)
results['Silh_best']=[None, (silh_best, n_cl_opt)]
results.to_csv('results.csv')