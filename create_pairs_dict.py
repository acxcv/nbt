import numpy as np
import torch
from word_vector import load_obj, save_obj
from retrieve_embedding_vectors import pad_embeddings_list,\
     retrieve_word_embeddings

embeddings_dict = load_obj('paragram-300-sl999')
candidate_pairs = load_obj('pairs_list')

candidate_pairs_dict = {}

for pair in candidate_pairs:
    pair_embeddings = []
    for word in pair:
        if ' ' in word:
            multi_word = word.split()
            word_embedding = [retrieve_word_embeddings(w, embeddings_dict)
                              for w in multi_word]
            word_embedding = np.array(word_embedding).sum(axis=0)
            word_embedding = torch.tensor(word_embedding)
        else:
            word_embedding = retrieve_word_embeddings(word, embeddings_dict)
        padded_word_embedding = pad_embeddings_list([word_embedding])
        pair_embeddings.append(torch.tensor(padded_word_embedding))
    candidate_pairs_dict[f'{pair}'] = pair_embeddings

save_obj(candidate_pairs_dict, 'pairs_dict')
