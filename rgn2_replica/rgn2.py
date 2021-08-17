# Author: Eric Alcaide ( @hypnopump ) 
import os
import sys
import torch
import numpy as np

from rgn2_replica.utils import *
from rgn2_replica.rgn2_utils import *


class RGN2_Naive(torch.nn.Module): 
    def __init__(self, layers=3, emb_dim=1280, hidden=256,
                 bidirectional=False, mlp_hidden=[32, 4], layer_type="LSTM"): 
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
    
    def forward(self, x, recycle=1, inter_recycle=False): 
        """ Inputs:
            * x: (B, L, Emb_dim) 
            * recycle: int. recycling iterations
            * recycle_iters: bool. whether to provide intermediate
                                   recycling iterations.
            Outputs:
            * x_pred: (B, L, 4). 
            * r_iters: list (recycle-1, B, L, 4)

            Note: 4 last dims of input is angles of previous point.
                  for first point, add dumb token [-5, -5, -5, -5]
        """
        r_iters = []
        x_buffer = x.clone() if recycle > 1 else x
        for i in range(recycle): 

            # reverse + move first angle token to last (angle padding)
            if self.bidirectional: 
                x_b = torch.flip(x_buffer.clone(), dims=(-2,))
                x_b = torch.cat([
                    x_b[..., :-4],
                    torch.cat([x_b[..., -1:, -4:], x_b[..., :-1, -4:]], dim=-2) 
                ], dim=-1)

            # forward and merge
            x_pred, (h_n, c_n) = self.stacked_lstm_f(x)
            if self.bidirectional: 
                x_b, (h_n_b, c_n_b) = self.stacked_lstm_b(x_b)
                x_pred = torch.cat([x_pred, x_b], dim=-1)

            x_pred = self.last_mlp(x_pred)

            # cat predictions to tokens for recycling
            if i < (recycle-1):
                x = torch.cat([ x_buffer[:, :, :-4], x_pred.detach() ], dim=-1)
                # store and return intermediate steps - only if not last
                if inter_recycle: 
                    r_iters.append(x_pred.detach())

        if inter_recycle: 
            r_iters = torch.stack(r_iters, dim=0)

        return x_pred, r_iters


    def predict_fold(self, x, recycle=1):
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

        # no gradients needed for prediction
        with torch.no_grad():

            r_policy = 1
            for i in range(x.shape[-2]):
                # only recycle (if set to) in last iter - saves time
                if i < ( x.shape[-2] - 1 ):
                    r_policy = recycle

                input_step = x[:, :i+1]              # (B, 0:i+1, 4)
                preds, r_iters = self.forward(       # (B, 1:i+2, 4)
                    input_step, recycle = r_policy
                ) 
                # only modify if it's not last iter. last angle is not needed
                if i < ( x.shape[-2] - 1 ):
                    x[:, 1:i+2, -4:] = preds

        # re-handles batch shape
        return preds.squeeze() if squeeze else preds, r_iters


class Refiner(torch.nn.Module): 
    """ Refines a protein structure by invoking several Rosetta scripts. """
    def __init__(self, **kwargs): 
        return

