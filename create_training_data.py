from nltk.util import pr
import pandas as pd
from word_vector import load_obj
from cnn import Rl_cnn
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

frame = load_obj("vector_frame")
frame = frame.head(10)
# pd.set_option('display.max_colwidth', None)
# print(frame.columns)
print(frame)
test_vector = frame["vectorized_utterance"].loc[0]
# test_tensor2 = frame["vectorized_utterance"].loc[1]
test_utterance = frame["tokenized_utterance"].loc[0]

# print("Original tensor shape: ", test_tensor.shape)
# print("Original tensor shape 2: ", test_tensor2.shape)

# print("original tensor: ", test_tensor)
# print('#################1', test_tensor)
# print(test_tensor.shape)
# print('#################2', test_tensor)
# print(test_tensor.shape)
# test_tensor = torch.unsqueeze(test_tensor, -1)
# print('#################3', test_tensor)
# print(test_tensor.shape)

# pad_tensor = torch.zeros(300, 206, 1)
# pad_tensor[:, :test_tensor.shape[1], :] = test_tensor

# print("Padded tensor: ", pad_tensor.shape)

def extract_ngram(embedding_list, n):
    b = 0
    e = n
    ngrams_out = []
    while e != len(embedding_list) + 1:
        concat_list = tuple(embedding_list[b:e])
        concatenated = np.concatenate((concat_list))
        ngrams_out.append(concatenated)
        b += 1
        e += 1
    
    return ngrams_out



def padded_tensor_from_ngram_list(ngram_list):
    ngram_input = torch.tensor(ngram_list)
    print("+++ tensor from ngram +++", ngram_input.shape)
    ngram_input = torch.transpose(ngram_input, 0, 1)
    print("+++ transposed tensor +++", ngram_input.shape)
    ngram_input = torch.unsqueeze(ngram_input, -1)
    print("+++ unsqueezed tensor +++", ngram_input.shape)
    padded = torch.zeros(900, 206, 1)
    padded[:ngram_input.shape[0], :ngram_input.shape[1], :] = ngram_input

    return padded
