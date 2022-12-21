import pandas as pd
import os
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
stopwords.words("english")
from ampligraph.evaluation import train_test_split_no_unseen
from Read_corpus import save_pkl, load_pkl, load_dict

def load_data(DATA_DIRECTORY: str, n: int):
    full_data=pd.DataFrame()
    i=0
    for filename in os.listdir(DATA_DIRECTORY):
        i+=1
        with open(DATA_DIRECTORY+'/'+filename, 'rb') as file:
            data = pickle.load(file)
            df=pd.DataFrame(data['triples'])
        full_data=full_data.append(df)
        if i>n:
            break
    print(str(i)+' documents read. Overall '+str(len(full_data))+' triples found')
    return full_data

def preprocess(text: str):
  text = re.sub(r"http\S+", "", text)
  text = re.sub(r"QUANT_._.", "QUANT", text)
  text = re.sub("[^A-Za-z]+", " ", text)
  text = text.lower().strip()
  return text

def prepare_data(full_data, split_ratio, preprocessed: bool):
    X=np.array(full_data.values)

    if preprocessed:
        full_data[1]=full_data[1].apply(lambda x: preprocess(x)) #leave stop words
    full_data=full_data.drop(full_data.index[full_data.apply(lambda x: x[2] in ['!None!'], axis=1)])
    full_data=full_data.drop(full_data.index[full_data.apply(lambda x: x[1] in [''], axis=1)])
    full_data=full_data.drop_duplicates()
    entities=np.array(list(set(full_data.values[:, 0]).union(full_data.values[:, 2])))
    relations=np.array(list(set(full_data.values[:, 1])))
    t_size=round(split_ratio*len(full_data))
    data = {}
    data['train'], data['valid'] = train_test_split_no_unseen(X, test_size=t_size, seed=1, allow_duplication=False) 
    data['train'], data['test']=train_test_split_no_unseen(data['train'], test_size=t_size, seed=1, allow_duplication=False)
    print('Unique entities: ', len(entities))
    print('Unique relations: ', len(relations))
    print('Train set size: ', data['train'].shape)
    print('Test set size: ', data['test'].shape)
    print('Valid set size: ', data['valid'].shape)
    return data, entities, relations

def to_dict(data, entities, relations):
    return {
        "entities": entities,
        "relations": relations,
        "train": data['train'],
        "test": data['test'],
        "valid": data['valid']
    }

def print_sizes(data, entities, relations):
    print('Unique entities: ', len(entities))
    print('Unique relations: ', len(relations))
    print('Train set size: ', data['train'].shape)
    print('Test set size: ', data['test'].shape)
    print('Valid set size: ', data['valid'].shape)

def descriptive_stat(data):
    full_data=np.concatenate([data['train'], data['valid'], data['test']], 0)
    df=pd.DataFrame(full_data)
    print("Top-5 most frequent relations:")
    print(df[1].value_counts().head(5))
    print("Relations' length distribution:")
    df['rel_length']=df[1].apply(lambda x: len(word_tokenize(x)))
    print(df['rel_length'].value_counts())

def Prepare_data_run(): 

    full_data=load_data('OPIEC_read', 3) # 3 - for subset 1; 9 - for subset 2; 100 for full data

    data, entities, relations = prepare_data(full_data, 0.1, True)
    to_pickle = to_dict(data, entities, relations)
    save_pkl('Subset_1', to_pickle)  #Subset1 #Subset2 # Full_data
    data, entities, relations= load_dict('Subset_1')
    print("Subset_1:")

    full_data=load_data('OPIEC_read', 9) # 3 - for subset 1; 9 - for subset 2; 100 for full data
    data, entities, relations = prepare_data(full_data, 0.1, True)
    to_pickle = to_dict(data, entities, relations)
    save_pkl('Subset_2', to_pickle)  #Subset1 #Subset2 # Full_data
    data, entities, relations= load_dict('Subset_2')
    print("Subset_2:")

    full_data=load_data('OPIEC_read', 100) # 3 - for subset 1; 9 - for subset 2; 100 for full data
    data, entities, relations = prepare_data(full_data, 0.1, True)
    to_pickle = to_dict(data, entities, relations)
    save_pkl('Full_data', to_pickle)  #Subset1 #Subset2 # Full_data
    data, entities, relations= load_dict('Full_data')
    print("Full_data:")
    print_sizes(data, entities, relations)
    descriptive_stat(data)


if __name__ == "__main__":
    
    #subset 1
    full_data=load_data('OPIEC_read', 3) # 3 - for subset 1; 9 - for subset 2; 100 for full data
    data, entities, relations = prepare_data(full_data, 0.1, True)
    to_pickle = to_dict(data, entities, relations)
    save_pkl('Subset_1', to_pickle)  #Subset1 #Subset2 # Full_data
    print_sizes(data, entities, relations)

    #subset 2
    full_data=load_data('OPIEC_read', 9) 
    data, entities, relations = prepare_data(full_data, 0.1, True)
    to_pickle = to_dict(data, entities, relations)
    save_pkl('Subset_2', to_pickle)  #Subset1 #Subset2 # Full_data
    print_sizes(data, entities, relations)

    #full data
    full_data=load_data('OPIEC_read', 100) # 3 - for subset 1; 9 - for subset 2; 100 for full data
    data, entities, relations = prepare_data(full_data, 0.1, True)
    to_pickle = to_dict(data, entities, relations)
    save_pkl('Full_data', to_pickle)  
    print_sizes(data, entities, relations)
    descriptive_stat(data)



