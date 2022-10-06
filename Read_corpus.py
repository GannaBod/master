from avro.datafile import DataFileReader
from avro.io import DatumReader
import os
import pickle

def save_pkl(path, file):
  with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def init_dict():
  sentences=[]
  triples=[]
  dictnr={'triples': triples, 'sentences':sentences}
  return dictnr

def extract_triples(reader: DataFileReader):
  # subjects=[]
  # relations=[]
  # objects=[]
  sentences=[]
  triples=[]

  for i, triple in enumerate(reader):
    subject=[]
    relation=[]
    t_object=[]
    sentence=[]
    #t_triple=[]
    #print(i)

    for i, element in enumerate(triple['subject']):
      subject.append(triple['subject'][i]['lemma'])
    for i, element in enumerate(triple['relation']):
      relation.append(triple['relation'][i]['lemma'])
    for i, element in enumerate(triple['object']):
      if triple['object'][i]['lemma'] is None:
         t_object.append("!None!")
      else: t_object.append(triple['object'][i]['lemma'])
    # objects.append(" ".join(t_object))
    #t_triple=((" ".join(subject)), (" ".join(t_object)), (" ".join(relation)))
    if "!None!" in (subject or relation or t_object):
      continue
    t_triple=[(" ".join(subject)), (" ".join(relation)),  (" ".join(t_object))]
    # subjects.append(" ".join(subject))
    # relations.append(" ".join(relation))
    # objects.append(" ".join(t_object))
    
    triples.append(t_triple)
    
    one_sent=triple['sentence_linked']
    for i, token in enumerate(one_sent['tokens']):
      sentence.append(token['word'])
    sentences.append(" ".join(sentence))
    #break
  #print(triples)
    
  return {'triples': triples, 'sentences':sentences}
  
def extract_triples_full_data(AVRO_SCHEMA_FILE, AVRO_DIRECTORY):
  full_data=init_dict()
  n=27
  i=0
  for filename in sorted(os.listdir(AVRO_DIRECTORY)):
    if filename=='_SUCCESS':
      continue
    #print(full_data)
    #print('Full data triples length: {}'.format(len(full_data['triples'])))
    i+=1
    print(filename)
    reader = DataFileReader(open(AVRO_DIRECTORY+"/"+filename, "rb"), DatumReader())
    my_data_dict=extract_triples(reader)
    print('New data triples length: {}'.format(len(my_data_dict['triples'])))
    full_data['triples'].extend(my_data_dict['triples'])
    full_data['sentences'].extend(my_data_dict['sentences'])
    print("Full data updated")
    print('Full data triples length new: {}'.format(len(full_data['triples'])))
    if i>100:
     save_pkl("./OPIEC_read"+str(n), full_data)
     n+=1
     full_data=init_dict()
     i=0
      #break
    reader.close()

  #return full_data
if __name__ == "__main__":

  AVRO_SCHEMA_FILE = "./avroschema/TripleLinked.avsc"
  AVRO_DIRECTORY="G:/My Drive/Colab Notebooks/data/OPIEC-linked-triples"
  full_data=extract_triples_full_data(AVRO_SCHEMA_FILE, AVRO_DIRECTORY)

