import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SemanticDecoder(nn.Module):
    def __init__(self):
        super(SemanticDecoder, self).__init__()

        self.linear_layer = nn.Linear(in_features=300, out_features=300,
                                      bias=True)

    def forward(self, user_utterance_representation,
                candidate_pair_representation):
        """
        :param user_utterance_representation: torch.tensor 300-dimensional
        representation vector of user utterance
        :param candidate_pair_representation: np.array -> contains two tensors
        with 300-dimensional vectors of a candidate pair
        :return: np.array TODO: DEFINE THIS PROPERLY
        """

        # pass the data through linear layer and do element-wise multiplication
        summed_candidate_rep = torch.add(candidate_pair_representation[0],
                                         candidate_pair_representation[1])
        linearized = self.linear_layer(summed_candidate_rep)
        sigmoid_out = torch.sigmoid(linearized)
        element_product = \
            np.multiply(user_utterance_representation.detach().numpy(),
                        sigmoid_out.detach().numpy())

        return element_product

