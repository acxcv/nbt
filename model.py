import torch
import torch.nn as nn
import torch.nn.functional as F


class Rl_cnn(nn.Module):

    def __int__(self):
        super(Rl_cnn, self).__int__()

        # define CNN parameters
        self.input_dimensions = 1
        self.output_size = 300

        # convolution layers
        self.unigram_conv_layer = nn.Conv1d(in_channels=self.input_dimensions,
                                            out_channels=self.output_size,
                                            kernel_size=1)
        self.bigram_conv_layer = nn.Conv1d(in_channels=self.input_dimensions,
                                           out_channels=self.output_size,
                                           kernel_size=2)
        self.trigram_conv_layer = nn.Conv1d(in_channels=self.input_dimensions,
                                            out_channels=self.output_size,
                                            kernel_size=3)

        # max pooling layers
        self.pool_uni = nn.MaxPool1d(kernel_size=1)
        self.pool_bi = nn.MaxPool1d(kernel_size=2)
        self.pool_tri = nn.MaxPool1d(kernel_size=3)

        # fully connected later
        # TODO:
        self.fc = nn.Linear()  #################### TODO

    def forward(self, x):
        x = self.unigram_conv_layer(x)
        x = torch.relu(x)
        x = self.pool
