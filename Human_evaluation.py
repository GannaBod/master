#load clusters.labels_ list from pickle file
import pandas as pd
from Read_corpus import load_pkl, load_dict

def evaluate_clusters(clusters, relations):
    d=pd.DataFrame({'relation':relations,'cluster':clusters})
    d=d.groupby(by='cluster').agg({'relation': lambda x: ", ".join(x)})
    d['list']=d['relation'].apply(lambda x: x.split(','))
    d['length']=d['list'].apply(lambda x: len(x))
    d=d.reset_index()

    results=pd.DataFrame(columns=['Cluster', 'Correct'])
    correct=[]
    for cluster in list(set(clusters)):
        rel_list=d['list'][d['cluster']==cluster].tolist()
        rel_list=rel_list[0]
        res = [(a, b) for idx, a in enumerate(rel_list) for b in rel_list[idx + 1:]]
        evaluation=[]
        for pair in res:
            print(pair)
            x=input("Correct?")
            evaluation.append(x)
            print(evaluation)

        rel_freq=evaluation.count('yes') / len(evaluation)
        correct.append(rel_freq)
        results=results.append({'Cluster':cluster, 'Correct': rel_freq}, ignore_index=True)
    results=results.append({'Cluster': 'All', 'Correct': sum(correct)/len(correct)}, ignore_index=True)
    results.to_csv('Human evaluation.csv')

if __name__ == "__main__":
    #clusters=load_pkl('clusters_TransE')
    #data, entitites, relations=load_dict('SubsetData_3')
    relations=['be','run','eat','do','make','enjoy']
    clusters=[0,0,1,1,1,0]
    evaluate_clusters(clusters, relations)
