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

from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE, ConvE, ConvKB, RandomBaseline
from ampligraph.evaluation import select_best_model_ranking
from ampligraph.utils import save_model, restore_model

import requests

from ampligraph.datasets import load_from_csv

from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import train_test_split_no_unseen

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score


import re
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
from ampligraph.utils import create_tensorboard_visualizations

#TODO change function cause it's copied
def display_aggregate_metrics(ranks):
    print('Mean Rank:', mr_score(ranks)) 
    print('Mean Reciprocal Rank:', mrr_score(ranks)) 
    print('Hits@1:', hits_at_n_score(ranks, 1))
    print('Hits@10:', hits_at_n_score(ranks, 10))
    print('Hits@100:', hits_at_n_score(ranks, 100))

full_data=pd.DataFrame()
i=0
for filename in os.listdir('./OPIEC_read'):
  i+=1
  with open('./OPIEC_read/'+filename, 'rb') as file:
    data = pickle.load(file)
    #why pandas? maybe numpy straight forward?
    df=pd.DataFrame(data['triples'])
  full_data=full_data.append(df)
  if i>3:
    break

X=np.array(full_data.values)
num_test = 2000

data = {}
data['train'], data['test'] = train_test_split_no_unseen(X, test_size=num_test, seed=0, allow_duplication=False) 
data['test'], data['valid']=train_test_split_no_unseen(data['test'], test_size=100, seed=0, allow_duplication=False)

entities=np.array(list(set(full_data.values[:, 0]).union(full_data.values[:, 2])))
relations=np.array(list(set(full_data.values[:, 1])))

print('Unique entities: ', len(entities))
print('Unique relations: ', len(relations))

print('Train set size: ', data['train'].shape)
print('Test set size: ', data['test'].shape)
print('Valid set size: ', data['valid'].shape)