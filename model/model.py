import torch
import torch.nn as nn


class IntegratedModel(nn.Module):
    """
    model structure:
    will use LSTM cells as basic units of the network;
    might require constant modification to see which structure works best.

    """

    def __init__(self, num_layers, num_features, num_hidden, output_size):
        super(IntegratedModel, self).__init__()
        self.lstm = nn.LSTMCell(num_features, num_hidden)
        self.num_layers = num_layers
        self.output_proj = nn.Linear(num_hidden, output_size)
        # # each cell has distinct weight;
        # self.lstm_lst = []
        # for i in range(num_layers):
        #
        #     new_lstm = nn.LSTMCell(num_features, num_hidden, bias=False)
        #     self.lstm_lst.append(new_lstm)

    def forward(self, input):
        """
        :param input: should have size: batch_size * num_layers * num_features
                a single input for each corresponding layer should be extracted out before processing;
        :return:
        """
        curr_hidden, curr_cell = self.lstm(input[:, 0, :].squeeze())
        total_hidden = curr_hidden.unsqueeze(1)  # insert one dimension, for indicating layers
        for i in range(1, self.num_layers):
            curr_input = input[:, i, :].squeeze()
            # .squeeze() is intended to remove the intermediate dimension

            curr_hidden, curr_cell = self.lstm(curr_input, (curr_hidden, curr_cell))
            torch.cat((total_hidden, curr_hidden.unsqueeze(1)), dim=1)  # concatenation dimension should be the
            # same with dimension of layers;
        # once the hidden state is acquired, should have a method to project to the desired result.
        # will use a linear layer along with softmax if necessary;
        # below is just an example; realizing preprocessing result might affect target;
        output = self.output_proj(curr_hidden)  # can replace with total_hidden
        return output