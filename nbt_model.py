import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import sampler
from word_vector import load_obj
import torch.optim as optim
from cnn import Rl_cnn
from semantic_decoding import SemanticDecoder
from context_modeling import ContextModeler


class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()

        self.conv_nn = Rl_cnn()
        self.semantic_dec = SemanticDecoder()
        self.context_mod = ContextModeler()

        # declare hidden layers
        # linear layer for the sum of mr, mc and d
        self.linear_layer_100_dim = nn.Linear(in_features=300,
                                              out_features=100,
                                              bias=True)
        # binary linear layer
        self.linear_layer_2_dim = nn.Linear(in_features=1,
                                            out_features=2,
                                            bias=True)
        # final softmax for binary decision
        self.softmax_layer = nn.Softmax(dim=0)
        # dim=0 normalizes values along axis 0


    def generate_representations(self, frame_row_dict):
        """
        :param frame_row_dict: The input dict sample from the dataloader
        :return: intermediate_reps: dict containing the intermediate reps
        """

        user_utterance = frame_row_dict['user_utt']
        sysout = frame_row_dict['sys_out']
        cand_pair = frame_row_dict['cand_pair']

        intermediate_reps = {}
        cnn_utt_rep = self.conv_nn(user_utterance.float())
        cnn_sysout_reps = []

        for item in sysout:
            if isinstance(item, list):
                item_rep = [self.conv_nn(x.float()) for x in item]
            elif isinstance(item, torch.Tensor):
                item_rep = self.conv_nn(item.float())
            cnn_sysout_reps.append(item_rep)

        cand_pair_reps = [self.conv_nn(x.float()) for x in cand_pair]

        intermediate_reps['cnn_utt_rep'] = cnn_utt_rep
        intermediate_reps['cnn_sysout_reps'] = cnn_sysout_reps
        intermediate_reps['cnn_cand_pair_reps'] = cand_pair_reps

        return intermediate_reps

    def forward(self, batch):
        """
        :param intermediate_dict: dict -> dictionary containing rep vectors
        :returns binary_label: bool -> binary label for prediction  
        """
        out_list = []
        for sample in batch:
            intermediate_dict = self.generate_representations(sample)

            cnn_utt_rep = intermediate_dict['cnn_utt_rep']
            cnn_sysout_reps = intermediate_dict['cnn_sysout_reps']
            cnn_cand_pair_reps = intermediate_dict['cnn_cand_pair_reps']

            similarity_metric_d = self.semantic_dec(cnn_utt_rep,
                                                    cnn_cand_pair_reps)

            context_mod_dict = self.context_mod(cnn_sysout_reps,
                                                cnn_cand_pair_reps,
                                                cnn_utt_rep)
            # TODO: MAYBE TRY SUMMING Ms BEFORE PASSING THROUGH LINEAR LAYER
            transformed_reps = []
            transformed_reps.append(self.linear_layer_100_dim(
                torch.tensor(similarity_metric_d)))
            for k, v in context_mod_dict.items():
                if len(v) != 0:
                    linear_out = self.linear_layer_100_dim(v[0])
                    transformed_reps.append(linear_out)
            # for v in context_mod_dict.values():
            #     for tensor in v:
            #         linear_out = self.linear_layer_m(tensor)
            #         transformed_reps.append(linear_out)
            # TODO: Does it make sense to unsqueeze this?
            binary_linear_input = torch.sum(torch.stack(transformed_reps)).unsqueeze(0)
            binary_linear_output = self.linear_layer_2_dim(binary_linear_input)
            softmax_ouptut = self.softmax_layer(binary_linear_output)

            out_list.append(softmax_ouptut)

            # Turn list of output tensors into list of binary labels
            predictions = []
            for tensor in out_list:
                if tensor[0] > tensor[1]:
                    predictions.append(1)
                else:
                    predictions.append(0)
            predictions = torch.tensor(predictions)

        return predictions


# X = frame[['user_utt','sys_out','cand_pair']] # data
# y = frame['binary_label'] # label
