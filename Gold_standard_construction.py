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

def get_synonyms(relation):
  synonyms = []

  for syn in wordnet.synsets(relation):
      for i in syn.lemmas():
          synonyms.append(i.name())
  return list(set(synonyms))

#if __name__ =='__main__': #exemplary trial of the function and then #main file with the execution of the whole pipeline
# print(get_synonyms('travel'))

#TODO make function and reuse it evry time... or just save to the pkl file.

full_data=pd.DataFrame()
i=0
for filename in os.listdir('./OPIEC_read'):
  i+=1
  with open('./OPIEC_read/'+filename, 'rb') as file:
    data = pickle.load(file)
    #why pandas? maybe numpy straight forward?
    df=pd.DataFrame(data['triples'])
  full_data=full_data.append(df)
  if i>2:
    break
print(full_data)

relations=np.array(list(set(full_data.values[:, 1])))
one_w_relations=[]
for relation in relations:
  if len(word_tokenize(relation))<=1:
    one_w_relations.append(relation)
print(len(one_w_relations))


random.seed=4
random.shuffle(one_w_relations)
test=one_w_relations[:30]

data=pd.DataFrame({'relation':test})
data['synonyms']=data['relation'].apply(lambda x: get_synonyms(x))
data.to_csv('rel_syns.csv')

relations=pd.DataFrame({'relations':list(relations)})

for verb in data['relation'].to_list():
  synonyms=get_synonyms(verb)
  print(verb)

  for synonym in synonyms:
    if synonym in (relations['relations'].to_list()):
      print(synonym)
  print('--------')