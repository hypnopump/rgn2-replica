# Author: Eric Alcaide ( @hypnopump ) 
import os
import sys
import torch
import numpy as np

from rgn2_replica.utils import *


class RGN2_Naive(torch.nn.Module): 
    def __init__(self, layers=3, emb_dim=1280, hidden=256,
                 bidirectional=False, mlp_hidden=[16, 4], layer_type="LSTM"): 
        """ RGN2 module which turns embeddings into a Cα trace.
            Inputs: 
            * layers: int. number of rnn layers
            * emb_dim: int. number of dimensions in the input
            * hidden: int or list of ints. hidden dim at each layer
            * bidirectional: bool. whether to use bidirectional LSTM
        """
        super(RGN2_Naive, self).__init__()
        hidden_eff = lambda x: x + x*int(bidirectional)
        layer_types = {
            "LSTM": torch.nn.LSTM,
            "GRU": torch.nn.GRU,
        }
        # store params
        self.layer_type = layer_type
        self.layers = layers
        self.hidden = [emb_dim]+hidden if isinstance(hidden, list) else \
                      [emb_dim] + [hidden]*layers
        self.bidirectional = bidirectional
        self.mlp_hidden = mlp_hidden
        # declare layers

        self.stacked_lstm_f = layer_types[self.layer_type](
            input_size = self.hidden[0],
            hidden_size = self.hidden[1],
            batch_first = True,
            bidirectional = False,
            num_layers = self.layers
        ) 
        # add backward lstm
        if self.bidirectional:
            self.stacked_lstm_b = layer_types[self.layer_type](
                input_size = self.hidden[0],
                hidden_size = self.hidden[1],
                batch_first = True,
                bidirectional = False,
                num_layers = self.layers
            ) 

        self.last_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_eff(self.hidden[-1]), self.mlp_hidden[0]),
            torch.nn.SiLU(),
            torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[-1]) 
        )
    
    def forward(self, x): 
        """ Inputs:
            * x (B, L, Emb_dim) 
            Outputs: (B, L, 4). 
            
            Note: 4 last dims of input is angles of previous point.
                  for first point, add dumb token [-5, -5, -5, -5]
        """
        x, (h_n, c_n) = self.stacked_lstm(x)

        if self.bidirectional: 
            # reverse + move first angle token to last (angle padding)
            x_b = torch.flip(x.clone(), dims=-2)
            x_b = torch.cat([
                x_b[..., :-4],
                torch.cat([x_b[..., -1:, :], x_b[..., :-1, :]], dim=-2) 
            ], dim=-1)
            # predict + merge
            x_b, (h_n_b, c_n_b) = self.stacked_lstm_b(x_b)
            x = torch.cat([x, x_b], dim=-2)

        return self.last_mlp(x)

    def predict_fold(self, x):
        """ Autoregressively generates the protein fold
            Inputs: ((B), L, Emb_dim) -> Outputs: ((B), L, 4). 

            Note: 4 last dims of input is dumb token for first res. 
                  Use same as in `.forward()` method.
        """
        # handles batch shape
        squeeze = len(x.shape) == 2
        if squeeze: 
            x = x.unsqueeze(dim=0)
        x_t = x.transpose(0, 1) # works on length, not batch

        preds_list = []
        for i in range(x.shape[-2]):
            if i == 0: 
                input_step = x[:, i:i+1]
            else: 
                # add angle predicton from prev steps as feats.
                input_step = torch.cat([
                    x[:, 1:i+1, :-4],               # normal feats
                    torch.cat(preds_list, dim=-2)   # angles of previous steps
                ], dim=-1)
                
            preds_list.append( self.forward(input_step)[:, -1:] )

        final_pred = torch.cat(preds_list, dim=0).transpose(0,1)

        # re-handles batch shape
        return final_pred.squeeze() if squeeze else final_pred


class Refiner(torch.nn.Module): 
    """ Refines a protein structure by invoking several Rosetta scripts. """
    def __init__(self, **kwargs): 
        return

