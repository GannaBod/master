# import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
import pandas as pd
import os
# import random
import pickle
import numpy as np
# import ampligraph

# from ampligraph.latent_features import TransE, ComplEx, DistMult, HolE, ConvE, ConvKB, RandomBaseline
# from ampligraph.evaluation import select_best_model_ranking
# from ampligraph.utils import save_model, restore_model
# from ampligraph.evaluation import evaluate_performance
# from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import train_test_split_no_unseen

from Read_corpus import save_pkl, load_pkl

def load_data(DATA_DIRECTORY: str, n: int):
    full_data=pd.DataFrame()
    i=0
    for filename in os.listdir(DATA_DIRECTORY):
        i+=1
        with open(DATA_DIRECTORY+'/'+filename, 'rb') as file:
            data = pickle.load(file)
            #why pandas? maybe numpy straight forward?
            df=pd.DataFrame(data['triples'])
        full_data=full_data.append(df)
        if i>n:
            break
    print(str(i)+' documents read. Overall '+str(len(full_data))+' triples found')
    return full_data

def prepare_data(full_data, split_ratio):#, valid_size, test_size):
    X=np.array(full_data.values)

    #valid and test size proportional to the data length
    t_size=round(split_ratio*len(full_data))

    data = {}
    data['train'], data['valid'] = train_test_split_no_unseen(X, test_size=t_size, seed=1, allow_duplication=False) 
    data['train'], data['test']=train_test_split_no_unseen(data['train'], test_size=t_size, seed=1, allow_duplication=False)

    entities=np.array(list(set(full_data.values[:, 0]).union(full_data.values[:, 2])))
    relations=np.array(list(set(full_data.values[:, 1])))

    print('Unique entities: ', len(entities))
    print('Unique relations: ', len(relations))

    print('Train set size: ', data['train'].shape)
    print('Test set size: ', data['test'].shape)
    print('Valid set size: ', data['valid'].shape)
    #print(data['train'])

    return data, entities, relations

def to_dict(data, entities, relation):
    return {
        "entities": entities,
        "relations": relations,
        "train": data['train'],
        "test": data['test'],
        "valid": data['valid']
    }



if __name__ == "__main__":

##CHOOSE THE DATA SUBSET
    full_data=load_data('G:/My Drive/Colab Notebooks/data/OPIEC', 100)

    data, entities, relations = prepare_data(full_data, 0.1) #, 100, 2000) #train test valid size

#save data for future use
    to_pickle = to_dict(data, entities, relations)
    save_pkl('All_data_final', to_pickle)
    # with open("SubsetData", 'wb') as handle:
    #     pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)