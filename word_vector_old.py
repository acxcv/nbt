# import time
import pandas as pd
import pickle

def load_vectors(embedding_file):


    embeddings = open(embedding_file, encoding='utf-8', errors="ignore")
    dictionary = {}
    for line in embeddings:
        values = line.split()
        word = values[0]
        embeddings = values[1:]
        dictionary[word] = embeddings
    return dictionary

def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

