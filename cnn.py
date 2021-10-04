import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Rl_cnn(nn.Module):

    def __init__(self):
        super(Rl_cnn, self).__init__()

        # define CNN parameters
        # DEPRECATED: self.input_dimensions = 206  # Maximum input (utterance) length
        self.embedding_size, self.output_size = 300, 300

        # convolution layers
        self.unigram_conv_layer = self.create_conv_layer(ngram_size=1)
        self.bigram_conv_layer = self.create_conv_layer(ngram_size=2)
        self.trigram_conv_layer = self.create_conv_layer(ngram_size=3)

    def create_conv_layer(self, ngram_size):
        conv_layer = nn.Conv2d(in_channels=1, 
                               out_channels=self.output_size,
                               kernel_size=(ngram_size,
                                            self.embedding_size), 
                               bias=True)
        return conv_layer

    def forward(self, input_sentence_matrix):
        """
        :param input_sentence_matrix: Matrix where each row represents one
        word from the sentence, and each column represents an embedding dim
        :return: Final utterance representation vector
        """

        pooled_1 = self.reduce_dimensions(input_sentence_matrix, 1)
        pooled_2 = self.reduce_dimensions(input_sentence_matrix, 2)
        pooled_3 = self.reduce_dimensions(input_sentence_matrix, 3)

        final_representation = pooled_1 + pooled_2 + pooled_3

        return final_representation
    

    def reduce_dimensions(self, sentence_matrix, ngram_size):
        # TODO: CHECK BATCH DIMENSION AND MAYBE DELETE THE FOLLOWING LINE
        x = sentence_matrix.unsqueeze(dim=0)
        x = x.unsqueeze(dim=1)
        if ngram_size == 1:
            x = self.unigram_conv_layer(x)
        elif ngram_size == 2:
            x = self.bigram_conv_layer(x)
        elif ngram_size == 3:
            x = self.trigram_conv_layer(x)
        x = x.squeeze(dim=-1)
        x = torch.relu(x)
        x = F.max_pool1d(x, kernel_size=(x.size(dim=2)))
        x = x.squeeze(dim=-1)

        return x


# tensor1 = torch.rand(40, 300)


# model = Rl_cnn()
# print(model(tensor1))
