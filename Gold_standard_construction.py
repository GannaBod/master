## Get random simple relations, extend with synonyms which appear in the data

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import pickle
import numpy as np
import random
import json
import re
#from Models_results import load_data
from Read_corpus import load_dict #load_pkl

def get_synonyms(relation):
  synonyms = []
  synonyms.append(relation)
  for syn in wordnet.synsets(relation):
      for i in syn.lemmas():
          synonyms.append(i.name())
  return list(set(synonyms))

def gold_st_json(path):
  with open(path) as f:
    gs=json.load(f)
  gs_seed=np.array(list(gs.keys()))
  gs_seed=[re.sub('_', ' ', rel) for rel in gs_seed]
  gs_df=pd.DataFrame.from_dict(gs)
#gs_df=gs_df.reset_index()
#gs_df=gs_df.drop([1], axis=0)
  gs_df=pd.DataFrame.transpose(gs_df)
  gs_df=pd.DataFrame({'verb': gs_df['rp']})
  gs_df['cluster']=np.arange(len(gs_df))
  gs_df=gs_df.explode('verb')
  gs_df.to_csv('rp_gold_standard.csv')
#gs_df #.reset_index().columns#=gs_df.explode('rp')

#TODO make function and reuse it evry time... or just save to the pkl file.

# full_data=pd.DataFrame()
# i=0
# for filename in sorted(os.listdir('./OPIEC_read')):
#   i+=1
#   with open('./OPIEC_read/'+filename, 'rb') as file:
#     data = pickle.load(file)
#     #why pandas? maybe numpy straight forward?
#     df=pd.DataFrame(data['triples'])
#   full_data=full_data.append(df)
#   if i>2:
#     break
#print(full_data)

def get_data(Path):
  data=load_dict(Path)
  # data['train']=data['train_set']
  #data['test']=data['test_set']
  # data['valid']=data['valid_set']
  # entities=data['entities']
  # relations=data['relations']
  #df=np.array(data['test'].values) #-> test triples. but relations are seen 
  relations=list(set(data['test'][:, 1]))
  return relations


def get_1w_rel(relations):
  one_w_relations=[]
  for relation in relations:
    if len(word_tokenize(relation))<=1:
      one_w_relations.append(relation)
  print("Number of 1-word relations in the subset:", len(one_w_relations)) #1187 #1335
  return one_w_relations

def rel2cluster(one_w_relations, n):
  random.seed(1)  
  random.shuffle(one_w_relations)
  test=one_w_relations[:n]
  data=pd.DataFrame({'relation':test})
  data['synonyms']=data['relation'].apply(lambda x: get_synonyms(x))
  data['cluster']= np.arange(len(data))
  data=data.explode('synonyms')
  data=data.drop(['relation'], axis=1)
  data=data.rename({'synonyms': 'verb'}, axis=1)
  data.to_csv('Gold_standard.csv')

#only relations and clusters are used in the evaluation... -> should I revise the sentences or not?
#def 
#cluster=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/data/gold_stand_cluster.csv")
# cluster_data=pd.DataFrame()
# for verb in data['synonyms']:
#   i = data['cluster'][data['verb']==verb].to_list()[0]
#   data_verb=full_data_df[['sentences', 'triples']][full_data_df['relations']==verb]
#   data_verb=data_verb.assign(verb=verb)
#   data_verb=data_verb.assign(cluster=i)
#   cluster_data=pd.concat([cluster_data, data_verb], ignore_index=True)

# def triples2ids(triple:list):
#   ids=[]
#   sub_id=entity_id_map.get(triple[0])
#   rel_id=relation_id_map.get(triple[1])
#   obj_id=entity_id_map.get(triple[2])
#   ids.append((sub_id, rel_id, obj_id))
#   return ids
# #cluster_data['triples_new']=cluster_data['triples'].apply(lambda x: eval(x))
# cluster_data['triples_id']=cluster_data['triples'].apply(lambda x: triples2ids(x))
# cluster_data.to_csv("Gold_standard.csv")
# cluster_data

def Gold_st_construction():
  data, entities, relations=load_dict('Subset_1')
  one_w_relations = get_1w_rel(relations)
  rel2cluster(one_w_relations, 30)


if __name__ == "__main__":

#phase 1: get synonyms

  #relations=get_data('Preprocessed')

  data, entities, relations=load_dict('Subset_3_docs')
  one_w_relations = get_1w_rel(relations)
  rel2cluster(one_w_relations, 30)     #GS ver 2


  #gold_st_json("C:/Users/b-ann/Documents/UNI MANNHEIM/Master Data Science/Master Thesis/rp_canonicalisation-main/rp_canonicalisation-main/dataset/162NormalisedRP.json")
#phase 2: get verbs and clusters dataframe
#phase 3: retrieve trriples with these verbs


# relations=pd.DataFrame({'relations':list(relations)})

# for verb in one_w_relations:
#   synonyms=get_synonyms(verb)
#   print(verb)

#   for synonym in synonyms:
#     if synonym in (relations):
#       print(synonym)
#   print('--------')

