# Author: Eric Alcaide ( @hypnopump ) 
import os
import sys
import numpy as np

import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNBase, LSTMCell
from torch.nn import functional as F

from rgn2_replica.utils import *


class AminoBERT(torch.nn.Module): 
    """ AminoBERT module providing high-quality embeddings for AAs. """
    def __init__(self, **kwargs): 
        return


class RGN2_LSTM_Naive(torch.nn.Module): 
    def __init__(self, layers, emb_dim=1280, hidden=[256, 128, 64], 
                 bidirectional=True, mlp_hidden=[32, 4]): 
        """ RGN2 module which turns embeddings into a Cα trace.
            Inputs: 
            * layers: int. number of rnn layers
            * emb_dim: int. number of dimensions in the input
            * hidden: int or list of ints. hidden dim at each layer
            * bidirectional: bool. whether to use bidirectional LSTM
        """
        super(RGN2_LSTM_Naive, self).__init__()
        hidden_eff = lambda x: x + x*int(bidirectional)
        # store params
        self.layers = layers
        self.hidden = [emb_dim]+hidden if isinstance(hidden, list) else \
                      [emb_dim] + [hidden]*layers
        self.bidirectional = bidirectional
        self.mlp_hidden = mlp_hidden
        # declare layers
        self.stacked_lstm = torch.nn.ModuleList([
            torch.nn.LSTM(input_size = hidden_eff(self.hidden[i]) if i!=0 else self.hidden[i],
                          hidden_size = self.hidden[i+1],
                          batch_first = True,
                          bidirectional = bidirectional,
                         ) for i in range(self.layers)
        ])
        self.last_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_eff(self.hidden[-1]), self.mlp_hidden[0]),
            torch.nn.SiLU(),
            torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[-1]) 
        )
    
    def forward(self, x): 
        """ Inputs: (..., Emb_dim) -> Outputs: (..., 4). """
        for i,rnn_layer in enumerate(self.stacked_lstm): 
            x, (h_n, c_n) = rnn_layer(x)
        return self.last_mlp(x)

    def predict_fold(self, x):
        """ Autoregressively generates the protein fold
            Inputs: (..., Emb_dim) -> Outputs: (..., 4). 
        """
        raise NotImplementedError


# from: https://florianwilhelm.info/2018/08/multiplicative_LSTM_for_sequence_based_recos/
class mLSTM(RNNBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(mLSTM, self).__init__(
            mode='LSTM', input_size=input_size, hidden_size=hidden_size,
                 num_layers=1, bias=bias, batch_first=True,
                 dropout=0, bidirectional=False)

        w_im = torch.Tensor(hidden_size, input_size)
        w_hm = torch.Tensor(hidden_size, hidden_size)
        b_im = torch.Tensor(hidden_size)
        b_hm = torch.Tensor(hidden_size)
        self.w_im = Parameter(w_im)
        self.b_im = Parameter(b_im)
        self.w_hm = Parameter(w_hm)
        self.b_hm = Parameter(b_hm)

        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        n_batch, n_seq, n_feat = input.size()

        hx, cx = hx
        steps = [cx.unsqueeze(1)]
        for seq in range(n_seq):
            mx = F.linear(input[:, seq, :], self.w_im, self.b_im) * F.linear(hx, self.w_hm, self.b_hm)
            hx = (mx, cx)
            hx, cx = self.lstm_cell(input[:, seq, :], hx)
            steps.append(cx.unsqueeze(1))

        return torch.cat(steps, dim=1)


class Refiner(torch.nn.Module): 
    """ Refines a protein structure by invoking several Rosetta scripts. """
    def __init__(self, **kwargs): 
        return



