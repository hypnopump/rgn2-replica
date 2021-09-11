# Author: Eric Alcaide ( @hypnopump ) 
import os
import sys
from typing import Optional, Tuple
# science
import numpy as np
# ML
import torch
import torch.nn.functional as F
from x_transformers import XTransformer, Encoder
from einops import rearrange, repeat
# custom
import mp_nerf
from rgn2_replica.utils import *


#######################
#### USEFUL PIECES ####
#######################

@torch.jit.script
def prediction_wrapper(x: torch.Tensor, pred: torch.Tensor):
    """ Facilitates recycling. Inputs the original input + prediction
        Returns a new original input. 
        This case is specific for this task, but could be adapted. 
        Inputs: 
        * x: (B, L, Emb_dim) float tensor. Emb dim incorporates already pred_dim
        * pred: (B, L, pred_dim)
        Outputs: (B, L, Emb_dim)
    """
    # ensure preds' first values
    preds[:, 0, [0, 2]] = 0.
    preds[:, 0, [1, 3]] = 1.
    preds[:, 1, 2] = 0.
    preds[:, 1, 3] = 1.
    # refill x with preds
    x_ = x.clone()
    x_[:, :-1, -pred.shape[-1]:] = pred.detach()
    return x_


def pred_post_process(points_preds: torch.Tensor, 
                      seq_list: Optional[List] = None,
                      mask: Optional[torch.Tensor] = None):
    """ Converts an angle-based output to structures. 
        Inputs:
        * points_preds: (B, L, 2, 2)
        * seq_list: (B,) list of str. FASTA sequences. Optional. build scns
        * mask: (B, L) bool tensor. 
        Outputs: 
        * ca_trace_pred: (B, L, 14, 3)
        * frames_preds: (B, L, 3, 3)
        * wrapper_pred: (B, L, 14, 3)
    """
    device = points_preds.device
    if mask is None:
        mask = torch.ones(points_preds.shape[:-2], dtype=torch.bool)
    # restate first values to known ones (1st angle, 1s + 2nd dihedral)
    points_preds[:, 0, [0, 1], 1] = 1.
    points_preds[:, 0, [0, 1], 0] = 0.
    points_preds[:, 1, 1, 1] = 1.
    points_preds[:, 1, 1, 0] = 0.
    
    # rebuild ca trace with angles - norm vectors to ensure mod=1. - (B, L, 14, 3)
    ca_trace_pred = torch.zeros(pooints_preds.shape[-2], 14, 3, device=device)              
    ca_trace_pred[:, :, 1], frames_preds = mp_nerf.proteins.ca_from_angles( 
        (points_preds / (points_preds.norm(dim=-1, keepdim=True) + 1e-7)).reshape(
            points_preds.shape[0], -1, 4
        )
    ) 
    ca_trace_pred = mp_nerf.utils.ensure_chirality(ca_trace_pred)
    
    # calc BB - can't do batched bc relies on extremes.
    wrapper_pred = torch.zeros_like(ca_trace_pred)
    for i in range(points_preds.shape[0]):
        wrapper_pred[i, mask[i]] = mp_nerf.proteins.ca_bb_fold( 
            ca_trace_pred[i:i+1, mask[i], 1] 
        )
        if seq_list is not None: 
            # build sidechains
            scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq=seq_list[i], device=device)
            wrapper_pred[i, mask[i]], _ = mp_nerf.proteins.sidechain_fold(
                wrapper_pred[i, mask[i]], **scaffolds, c_beta="backbone"
            )

    return points_preds, ca_trace_pred, frames_preds, wrapper_pred


# adapt LSTM to batch api
class LSTM(torch.nn.modules.rnn.RNNBase):
    def __init__(self, input_size, hidden_size, bias=True, num_layers=1, 
                 batch_first=True, dropout=0, bidirectional=False):
        """ Custom LSTM layer which supports batching by mask
            * input_size: read pytorch docs - LSTM
            * hidden_size: read pytorch docs - LSTM
            * bias: read pytorch docs - LSTM
            * num_layers: int. number of layers. only supports 1 for now. 
            * batch_first: bool. input should be (B, L, D) if True, 
                                (L, B, D) if False
            * dropout: float. amount of dropout to add to inputs. 
                              Not supported
            * bidirectional: bool. whether layer is bidirectional. Not supported. 
        """
        super().__init__(
            mode='LSTM', input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers, bias=bias, batch_first=batch_first,
                 dropout=dropout, bidirectional=bidirectional)

        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm_cell = torch.nn.modules.rnn.LSTMCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_: torch.Tensor, 
        hx_ : Optional[torch.Tensor]=None, 
        cx_ : Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Accepts inputs of variable length and resolves by masking. 
            ~Assumes sequences are ordered by descending length.~
        """
        device = input_.device
        n_batch, n_seq, n_feat = input_.size()

        # same as in https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
        if hx_ is None:
            hx_ = torch.zeros(input_.size(0), self.hidden_size, dtype=input_.dtype, 
                                device=device)
        if cx_ is None: 
            cx_ = torch.zeros(input_.size(0), self.hidden_size, dtype=input_.dtype, 
                                device=device)
        if mask is None: 
            mask = torch.ones(input_.shape[:-1], dtype=torch.bool, device=device)
        
        steps = []
        unbind_dim = 1 if self.batch_first else 0
        for seq, mask_ in zip(input_.unbind(unbind_dim), mask.unbind(unbind_dim)):
            for k in range(self.num_layers):
                # select appropiate by masking
                masked_input = seq[mask_] # (B, D) -> (D,)
                masked_hx, masked_cx = hx_[mask_], cx_[mask_]

                # pass
                hx_[mask_], cx_[mask_]  = self.lstm_cell(masked_input, (masked_hx, masked_cx))

                # record hiddens
                steps.append(hx_.clone())
                
        outs = torch.stack(steps, dim=1)
        if not self.batch_first:
            outs = outs.transpose(0,1)
            
        return outs, (hx_, cx_)


##############
### MODELS ###
##############


class RGN2_Transformer(torch.nn.Module): 
    def __init__(self, embedding_dim=1280, hidden=[512], mlp_hidden=[128, 4],
                 act="silu", x_transformer_config={
                    "depth": 8,
                    "heads": 4, 
                    "attn_dim_head": 64,
                    # "attn_num_mem_kv": 16, # 16 memory key / values
                    "use_scalenorm": True, # set to true to use for all layers
                    "ff_glu": True, # set to true to use for all feedforwards
                    "attn_collab_heads": True,
                    "attn_collab_compression": .3,
                    "cross_attend": False,
                    "gate_values": True,  # gate aggregated values with the input"
                    # "sandwich_coef": 6,  # interleave attention and feedforwards with sandwich coefficient of 6
                    "rotary_pos_emb": True  # turns on rotary positional embeddings"
                 }
        ): 
        """ Transformer drop-in for RGN2-LSTM.
            Inputs: 
            * layers: int. number of rnn layers
            * mlp_hidden: list of ints. 
        """
        super(RGN2_Transformer, self).__init__()
        act_types = {
            "relu": torch.nn.ReLU, 
            "silu": torch.nn.SiLU,
        }
        # store params
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.mlp_hidden = mlp_hidden

        # declare layers
        """ Declares an XTransformer model.
            * No decoder, just predict embeddings
            * project with a lst_mlp

        """
        self.to_latent = torch.nn.Linear(self.embedding_dim, self.hidden[0])
        self.transformer = Encoder(
            dim= self.hidden[-1],

            **x_transformer_config
        )
        self.last_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden[-1], self.mlp_hidden[0]),
            act_types[act](),
            torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[-1]) 
        )

    
    def forward(self, x, mask : Optional[torch.Tensor] = None, 
                     recycle:int = 1, inter_recycle:bool = False): 
        """ Inputs:
            * x (B, L, Emb_dim) 
            Outputs: (B, L, 4). 
            
            Note: 4 last dims of input is angles of previous point.
                  for first point, add dumb token [-5, -5, -5, -5]
        """
        # same input for both rgn2-stm and transformer, so mask angles
        r_iters = []
        x_buffer = x.clone() if recycle > 1 else x   # buffer for recycling
        x[..., -4:] = 0.

        for i in range(max(1, recycle)): 
            x_pred = self.to_latent(x)
            x_pred = self.transformer(x_pred, mask=mask)
            x_pred = self.last_mlp(x_pred)

            # cat predictions to tokens for recycling
            if i < recycle:
                # normalize angles to avoid unstability
                angles = x_pred.detach()[:, :-1].reshape(x.shape[0], -1, 2, 2)
                angles = F.normalize(angles, dim=-1).reshape(x.shape[0], -1, 4)
                # regen inputs
                x = prediction_wrapper(x_buffer, angles)
                # store and return intermediate steps - only if not last
                if inter_recycle: 
                    r_iters.append(x_pred.detach())

        r_iters = torch.stack(r_iters, dim=-3) if inter_recycle else \
                  torch.empty(x.shape[0], recycle-1, device=x.device) # (B, recycle-1, L, 4)

        return x_pred, r_iters


    def predict_fold(self, x, mask : Optional[torch.Tensor] = None,
                          recycle:int = 1, inter_recycle:bool = False):
        """ Predicts all angles at once so no need for AR prediction. 
            Same inputs / outputs than  
        """
        with torch.no_grad(): 
            return self.forward(
                x=x, mask=mask, 
                recycle=recycle, inter_recycle=inter_recycle
            )


class RGN2_Naive(torch.nn.Module): 
    def __init__(self, layers=3, emb_dim=1280, hidden=256,
                 bidirectional=False, mlp_hidden=[32, 4], layer_type="LSTM",
                 act="silu", input_dropout=0.0, angularize=False): 
        """ RGN2 module which turns embeddings into a Cα trace.
            Inputs: 
            * layers: int. number of rnn layers
            * emb_dim: int. number of dimensions in the input
            * hidden: int or list of ints. hidden dim at each layer
            * bidirectional: bool. whether to use bidirectional LSTM
            * mlp_hidden: list of ints. dims for final MLP dimensions
            * layer_type: str. options present in `self.layer_types`
            * act: str. options present in `self.act_types`
            * input_dropout: float. dropout applied before all recurrent 
                                    layers independently
            * angularize: bool. whether to do single-value regression (False)
                                or predict a set of alphabet torsions (True).
        """
        super(RGN2_Naive, self).__init__()
        hidden_eff = lambda x: x + x*int(bidirectional)
        layer_types = {
            "LSTM": LSTM, # torch.nn.LSTM,
            "GRU": torch.nn.GRU,
        }
        act_types = {
            "relu": torch.nn.ReLU, 
            "silu": torch.nn.SiLU,
        }
        # store params
        self.layer_type = layer_type
        self.num_layers = layers
        self.hidden = [emb_dim]+hidden if isinstance(hidden, list) else \
                      [emb_dim] + [hidden]*layers
        self.bidirectional = bidirectional
        self.mlp_hidden = mlp_hidden
        self.angularize = angularize
        # declare layers

        self.dropout = input_dropout # could use `Dropout2d`
        self.dropout_l = torch.nn.Dropout(p=self.dropout) if input_dropout else \
                         torch.nn.Identity() 

        self.stacked_lstm_f = torch.nn.ModuleList([
            layer_types[self.layer_type](
                # double size of input (cat of lstm_f, lstm_b) if not first layer
                input_size = hidden_eff(self.hidden[i]) if i!= 0 else self.hidden[i], 
                hidden_size = self.hidden[1],
                batch_first = True,
                bidirectional = False,
                num_layers = 1,
            ) for i in range(layers)
        ])
        # add backward lstm
        if self.bidirectional:
            self.stacked_lstm_b = torch.nn.ModuleList([
                layer_types[self.layer_type](
                    # double size of input (cat of lstm_f, lstm_b) if not first layer
                    input_size = hidden_eff(self.hidden[i]) if i!= 0 else self.hidden[i], 
                    hidden_size = self.hidden[1],
                    batch_first = True,
                    bidirectional = False,
                    num_layers = 1,
                ) for i in range(layers) 
            ])

        # jit-COMPILE if custom LSTM
        if isinstance(self.stacked_lstm_f, LSTM):
            self.stacked_lstm_f = torch.nn.ModuleList([
                torch.jit.script(self.stacked_lstm_f[i]) for i in range(self.num_layers)
            ])
            self.stacked_lstm_b = torch.nn.ModuleList([
                torch.jit.script(self.stacked_lstm_b) for i in range(self.num_layers)
            ])


        self.last_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_eff(self.hidden[-1]), self.mlp_hidden[0]),
            act_types[act](),
            torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[-1]) 
        )

        # declare infra needed for angularization level
        if self.angularize: 
            self.angs = torch.nn.Parameter( # init custom param to -pi, pi
                (2*torch.rand(1, 1, 2, self.mlp_hidden[-1]//2) - 1) * np.pi # final pred / 2 
            ) 
            self.register_parameter("angles", self.angs)


    def forward(self, x:torch.Tensor, mask: Optional[torch.Tensor] = None,
                     recycle:int = 1, inter_recycle:bool = False): 
        """ Inputs:
            * x: (B, L, Emb_dim) 
            * mask: ((B), L) bool. whether to predict point. 
            * recycle: int. recycling iterations
            * inter_recycle: bool. whether to provide intermediate
                                   recycling iterations.
            * input_dropout: float. dropout quantity at input. 
            Outputs:
            * x_pred: (B, L, 4). 
            * r_iters: list (recycle-1, B, L, 4)

            Note: 4 last dims of input is angles of previous point.
                  for first point, add dumb token [-5, -5, -5, -5]
        """
        r_iters = []
        x_buffer = x.clone() if recycle > 1 else x       # buffer for iters
        if mask is None: 
            seq_lens = torch.tensor([x.shape[1]]*x.shape[0], dtype=torch.long)
        else: 
            seq_lens = mask.sum(dim=-1).long() 

        for i in range( max(1, recycle) ): 
            # do N layers, cat directions between them
            x_pred = x.clone() if self.num_layers > 1 else x # buffer for layers

            for k in range(self.num_layers): 
                x_f, (h_n, c_n) = self.stacked_lstm_f[k]( 
                    self.dropout_l(x_pred) , mask=mask
                )

                if self.bidirectional: 
                    # reverse - only the sequence part
                    x_b = x_pred.clone()
                    for l, length in enumerate(seq_lens): 
                        x_b[l, :length] = torch.flip(x_b[l, :length], dims=(-2,))

                    # back pass
                    x_b, (h_n_b, c_n_b) = self.stacked_lstm_b[k]( 
                        self.dropout_l(x_b), mask=mask 
                    )
                    # reverse again to match forward direction
                    for l, length in enumerate(seq_lens): 
                        x_b[l, :length] = torch.flip(x_b[l, :length], dims=(-2,))
                    # merge w/ forward direction
                    x_pred = torch.cat([x_f, x_b], dim=-1)
                else: 
                    x_pred = x_f

            x_pred = self.last_mlp(x_pred)
            if self.angularize: 
                x_pred = self.turn_to_angles(x_pred)

            # cat predictions to tokens for recycling
            if i < recycle:
                # normalize angles to avoid unstability
                angles = x_pred.detach()[:, :-1].reshape(x.shape[0], -1, 2, 2)
                angles = F.normalize(angles, dim=-1).reshape(x.shape[0], -1, 4)
                # regen inputs
                x = prediction_wrapper(x_buffer, angles)
                # store and return intermediate steps - only if not last
                if inter_recycle: 
                    r_iters.append(x_pred.detach())
 
        r_iters = torch.stack(r_iters, dim=-3) if inter_recycle else \
                  torch.empty(x.shape[0], recycle-1, device=x.device) # (B, recycle-1, L, 4)

        return x_pred, r_iters


    def predict_fold(self, x, mask : Optional[torch.Tensor] = None, 
                           recycle : int = 1, inter_recycle : bool = False):
        """ Autoregressively generates the protein fold
            Inputs: 
            * x: ((B), L, Emb_dim) 
            * mask: ((B), L) bool. whether to predict sequence. 
            * recycle: int. recycling iterations
            * inter_recycle: bool. whether to provide intermediate
                                   recycling iterations.
            
            Outputs: 
            * x_pred: ((B), L, 4)
            * r_iters: list (recycle-1, B, L, 4)

            Note: 4 last dims of input is dumb token for first res. 
                  Use same as in `.forward()` method.
        """
        # default mask is everything
        if mask is None: 
            mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        # handles batch shape
        squeeze = len(x.shape) == 2
        if squeeze: 
            x = x.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)

        # no gradients needed for prediction
        with torch.no_grad():

            r_policy = 1
            for i in range(x.shape[-2]):
                # only recycle (if set to) in last iter - saves time
                if i < ( x.shape[-2] - 1 ):
                    r_policy = recycle

                input_step = x[mask[:, i], :i+1]        # (B, 0:i+1, 4)
                preds, r_iters = self.forward(       # (B, 1:i+2, 4)
                    input_step, recycle = r_policy, inter_recycle=inter_recycle
                ) 
                # only modify if it's not last iter. last angle is not needed
                if i < ( x.shape[-2] - 1 ):
                    x[mask[:, i], 1:i+2, -4:] = preds

        # re-handles batch shape
        return preds.squeeze() if squeeze else preds, r_iters


    def turn_to_angles(self, preds, angles=2): 
        """ Turns a softmax prediction (B, L, N*angles) -> (B, L, 2*angles). """
        probs = F.softmax( rearrange(preds, "b l (a n) -> b l a n", a=angles), dim=-1 ) # (B, L, angles, N)
        angles = mp_nerf.utils.circular_mean(angles=self.angles, weights=probs) # (B, L, angles)
        angles = mp_nerf.ml_utils.angle_to_point_in_circum(angles) # (B, L, angles, 2)
        return rearrange(angles, "b l a n -> b l (a n)")           # (B, L, 2 * angles)



class Refiner(torch.nn.Module): 
    """ Refines a protein structure by invoking several Rosetta scripts. """
    def __init__(self, **kwargs): 
        return
