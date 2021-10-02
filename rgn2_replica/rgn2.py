# Author: Eric Alcaide ( @hypnopump ) 
import os
import sys
from typing import Optional, Tuple, List
from functools import partial
# science
import numpy as np
# ML
import torch
import torch.nn.functional as F
from x_transformers import XTransformer, Encoder
from einops import rearrange, repeat
# custom
from rgn2_replica import mp_nerf
from rgn2_replica.utils import *
import en_transformer


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
    pred[:, 1, 2] = 1.
    pred[:, 1, 3] = 0.
    # refill x with preds
    x_ = x.clone()
    x_[:, 1:-1, -pred.shape[-1]:] = pred[:, 1:].detach()
    return x_


def pred_post_process(points_preds: torch.Tensor,
                      seq_list: Optional[List] = None,
                      mask: Optional[torch.Tensor] = None,
                      model = None,
                      refine_args = {}):
    """ Converts an angle-based output to structures.
        Inputs:
        * points_preds: (B, L, 2, 2)
        * seq_list: (B,) list of str. FASTA sequences. Optional. build scns
        * mask: (B, L) bool tensor.
        * model: subclass of torch.nn.Module. prediction model w/ potential refiner
        * model_args: dict. arguments to pass to model for refinement
        Outputs:
        * ca_trace_pred: (B, L, 14, 3)
        * frames_preds: (B, L, 3, 3)
        * wrapper_pred: (B, L, 14, 3)
    """
    device = points_preds.device
    if mask is None:
        mask = torch.ones(points_preds.shape[:-2], dtype=torch.bool)
    lengths = mask.sum(dim=-1).cpu().detach().tolist()
    # restate first values to known ones (1st angle, 1s + 2nd dihedral)
    points_preds[:, 0, [0, 1], 0] = 1.
    points_preds[:, 0, [0, 1], 1] = 0.
    points_preds[:, 1, 1, 0] = 1.
    points_preds[:, 1, 1, 1] = 0.

    # rebuild ca trace with angles - norm vectors to ensure mod=1. - (B, L, 14, 3)
    ca_trace_pred = torch.zeros(*points_preds.shape[:-2], 14, 3, device=device)
    ca_trace_pred[:, :, 1], frames_preds = mp_nerf.proteins.ca_from_angles(
        (points_preds / (points_preds.norm(dim=-1, keepdim=True) + 1e-7)).reshape(
            points_preds.shape[0], -1, 4
        )
    )
    # delete extra part and chirally reflect
    ca_trace_pred_aux = torch.zeros_like(ca_trace_pred)
    for i in range(points_preds.shape[0]): 
        ca_trace_pred_aux[i, :lengths[i]] = ca_trace_pred_aux[i, :lengths[i]] + \
                                            mp_nerf.utils.ensure_chirality(ca_trace_pred[i:i+1, :lengths[i]])
    ca_trace_pred = ca_trace_pred_aux

    # use model's refiner if available
    if model is not None:
        if model.refiner is not None:
            for i in range(mask.shape[0]):
                adj_mat = torch.from_numpy(
                    np.eye(lengths[i], k=1) + np.eye(lengths[i], k=1).T
                ).bool().to(device).unsqueeze(0)

                coors = ca_trace_pred[i:i+1, :mask[i].shape[-1], 1].clone()
                coors = coors.detach() if model.refiner.refiner_detach else coors
                feats, coors, r_iters = model.refiner(
                    feats=refine_args[model.refiner.feats_inputs][i:i+1, :lengths[i]], # embeddings
                    coors=coors,
                    adj_mat=adj_mat,
                    recycle=refine_args["recycle"],
                    inter_recycle=refine_args["inter_recycle"],
                )
                ca_trace_pred[i:i+1, :lengths[i], 1] = coors

    # calc BB - can't do batched bc relies on extremes.
    wrapper_pred = torch.zeros_like(ca_trace_pred)
    for i in range(points_preds.shape[0]):
        wrapper_pred[i, :lengths[i]] = mp_nerf.proteins.ca_bb_fold( 
            ca_trace_pred[i:i+1, :lengths[i], 1] 
        )
        if seq_list is not None:
            # build sidechains
            scaffolds = mp_nerf.proteins.build_scaffolds_from_scn_angles(seq=seq_list[i], device=device)
            wrapper_pred[i, :lengths[i]], _ = mp_nerf.proteins.sidechain_fold(
                wrapper_pred[i, :lengths[i]], **scaffolds, c_beta="backbone"
            )

    return points_preds, ca_trace_pred, frames_preds, wrapper_pred


class SqReLU(torch.jit.ScriptModule):
    r""" Squared ReLU activation from https://arxiv.org/abs/2109.08668v1. """

    def __init__(self):
        super().__init__()
        return

    @torch.jit.script_method
    def forward(self, x):
        """ Inputs (B, L, C) --> Outputs (B, L, C). """
        return F.relu(x)**2


# from: https://github.com/nmaac/acon/blob/main/acon.py
# adapted for MLP and scripted.
class AconC(torch.jit.ScriptModule):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width):
        super(AconC, self).__init__()
        self.p1 = torch.nn.Parameter(torch.randn(width))
        self.p2 = torch.nn.Parameter(torch.randn(width))
        self.beta = torch.nn.Parameter(torch.ones(width))

    @torch.jit.script_method
    def forward(self, x):
        """ Inputs (B, L, C) --> Outputs (B, L, C). """
        p1, p2, beta = self.p1, self.p2, self.beta
        while x.dim() > p1.dim():
            p1 = p1.unsqueeze(0)
            p2 = p2.unsqueeze(0)
            beta = beta.unsqueeze(0)
        return (p1 * x - p2 * x) * torch.sigmoid(beta * (p1 * x - p2 * x)) + p2 * x


# from https://github.com/FlorianWilhelm/mlstm4reco/blob/master/src/mlstm4reco/layers.py
# adapted to match pytorch's implementation
# from https://github.com/FlorianWilhelm/mlstm4reco/blob/master/src/mlstm4reco/layers.py
# adapted to match pytorch's implementation
class mLSTM(torch.nn.modules.rnn.RNNBase):
    def __init__(self, input_size, hidden_size, bias=True, num_layers=1,
                 batch_first=True, dropout=0, bidirectional=False, peephole=False):
        """ Multiplicative LSTM layer which supports batching by mask
            * input_size: read pytorch docs - LSTM
            * hidden_size: read pytorch docs - LSTM
            * bias: read pytorch docs - LSTM
            * num_layers: int. number of layers. only supports 1 for now.
            * batch_first: bool. input should be (B, L, D) if True,
                                (L, B, D) if False
            * dropout: float. amount of dropout to add to inputs.
                              Not supported
            * bidirectional: bool. whether layer is bidirectional. Not supported.
            * peephole: bool. whether to add peephole connections ( as the
                              original Schmidhuber paper:
                              http://www.bioinf.jku.at/publications/older/2604.pdf )
        """
        super().__init__(
            mode='LSTM', input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers, bias=bias, batch_first=batch_first,
                 dropout=dropout, bidirectional=bidirectional)

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.peephole = peephole

        w_im = torch.Tensor(self.num_layers, hidden_size, input_size)
        w_hm = torch.Tensor(self.num_layers, hidden_size, hidden_size)
        b_im = torch.Tensor(self.num_layers, hidden_size)
        b_hm = torch.Tensor(self.num_layers, hidden_size)
        self.weight_ih_l = torch.nn.Parameter(w_im) # input - hidden (forget)
        self.weight_hh_l = torch.nn.Parameter(w_hm) # hidden - hidden (update)
        self.bias_ih_l = torch.nn.Parameter(b_im)   # input - hidden (forget)
        self.bias_hh_l = torch.nn.Parameter(b_hm)   # hidden - hidden (update)

        if self.peephole:
            self.lstm_cell = PeepLSTMCell(input_size, hidden_size, bias)
        else:
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
                masked_hx = F.linear(masked_input, self.weight_ih_l[k], self.bias_ih_l[k]) * \
                            F.linear(masked_hx, self.weight_hh_l[k], self.bias_hh_l[k])
                hx_[mask_], cx_[mask_]  = self.lstm_cell(masked_input, (masked_hx, masked_cx))

                # record hiddens
                steps.append(hx_.clone())

        outs = torch.stack(steps, dim=1)
        if not self.batch_first:
            outs = outs.transpose(0,1)

        return outs, (hx_, cx_)


# adapt LSTM to same api
class LSTM(torch.nn.modules.rnn.RNNBase):
    def __init__(self, input_size, hidden_size, bias=True, num_layers=1,
                 batch_first=True, dropout=0, bidirectional=False, peephole=False):
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
            * peephole: bool. whether to add peephole connections ( as the
                              original Schmidhuber paper:
                              http://www.bioinf.jku.at/publications/older/2604.pdf )
        """
        super().__init__(
            mode='LSTM', input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers, bias=bias, batch_first=batch_first,
                 dropout=dropout, bidirectional=bidirectional)

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.peephole = peephole

        if self.peephole:
            self.lstm_cell = PeepLSTMCell(input_size, hidden_size, bias)
        else:
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

# fadapted rom https://discuss.pytorch.org/t/peephole-lstm-cell-implementation/116531
# no support for multiple layers
class PeepLSTMCell(torch.jit.ScriptModule):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size))

        self.weight_ch_i = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ch_f = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ch_o = torch.nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameter()

    @torch.jit.unused
    def reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    @torch.jit.script_method
    def forward(self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        hx, cx = state
        xh = ( torch.mm(input, self.weight_ih.t()) + self.bias_ih + \
               torch.mm(hx, self.weight_hh.t()) + self.bias_hh)

        i, f, _c, o = xh.chunk(4, 1)

        i = torch.sigmoid(i + (self.weight_ch_i * cx))
        f = torch.sigmoid(f + (self.weight_ch_f * cx))
        _c = torch.tanh(_c)

        cy = (f * cx) + (i * _c)

        o = torch.sigmoid(o + (self.weight_ch_o * cy))
        hy = o * torch.tanh(cy)

        return hy, cy



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
                 act="silu", input_dropout=0.0, angularize=False, refiner_args={}):
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
            * refiner_args: dict. arguments for a global refiner. empty if no ref.
        """
        super(RGN2_Naive, self).__init__()
        hidden_eff = lambda x: x + x*int(bidirectional)
        layer_types = {
            "LSTM": LSTM, # torch.nn.LSTM,
            "GRU": torch.nn.GRU,
            "mLSTM": mLSTM,
            "peepLSTM": partial(LSTM, peephole=True),
            "peepmLSTM": partial(mLSTM, peephole=True),
        }
        act_types = {
            "relu": torch.nn.ReLU,
            "silu": torch.nn.SiLU,
            "aconc": AconC,
            "relu_square": SqReLU,
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

        # jit-COMPILE if mLSTM or custom LSTM
        if isinstance(self.stacked_lstm_f, mLSTM) or isinstance(self.stacked_lstm_f, LSTM):
            self.stacked_lstm_f = torch.nn.ModuleList([
                torch.jit.script(self.stacked_lstm_f[i]) for i in range(self.num_layers)
            ])
            self.stacked_lstm_b = torch.nn.ModuleList([
                torch.jit.script(self.stacked_lstm_b) for i in range(self.num_layers)
            ])


        self.last_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_eff(self.hidden[-1]), self.mlp_hidden[0]),
            act_types[act]() if act != "aconc" else act_types[act](width=self.mlp_hidden[0]),
            torch.nn.Linear(self.mlp_hidden[0], self.mlp_hidden[-1])
        )

        # declare infra needed for angularization level
        if self.angularize:
            self.angs = torch.nn.Parameter( # init custom param to -pi, pi
                (2*torch.rand(1, 1, 2, self.mlp_hidden[-1]//2) - 1) * np.pi # final pred / 2
            )
            self.register_parameter("angles", self.angs)

        # init forget gates to open
        self.apply(self.init_)

        # potential global refiner
        self.refiner = None
        if len(refiner_args) >= 1:
            self.refiner = RGN2_Refiner_Wrapper(**refiner_args)


    def init_(self, module):
        """ initialize biases of LSTM forget gates to 1. so gradients flow initially
            from: http://proceedings.mlr.press/v37/jozefowicz15.pdf
            Recommends init to 1-3

            pytorch implements 2x biases - b_ih + b_hh -> 1+1 = 2.
        """
        if type(module) in {torch.nn.LSTM}:
            # ONLY WORKS FOR 1 LAYER - LSTM stores in tuple, not in stacked tensor
            shape = module.bias_ih_l0.shape[-1]
            torch.nn.init.constant_(module.bias_ih_l0[shape//4:shape//2], 1.)
            torch.nn.init.constant_(module.bias_hh_l0[shape//4:shape//2], 1.)
        elif type(module) in {mLSTM, LSTM}:
            shape = module.lstm_cell.bias_ih.shape[-1]
            # torch.nn.init.constant_(module.lstm_cell.bias_ih[ shape//4 : shape//2 ], 1.)
            # torch.nn.init.constant_(module.lstm_cell.bias_hh[ shape//4 : shape//2 ], 1.)


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


    def load_my_state_dict(self, state_dict, verbose=True):
        """ Loads a model from state dict.
            Tolerates both
            * in-model + out-of-dict
            * in-dict + out-of-model
        """
        own_state = self.state_dict()
        in_model_out_of_dict = set(own_state.keys())
        in_dict_out_of_model = set([])

        # solve naming with `_l0` and `_l`
        # adding = {}
        # for name, param in state_dict.items():
        #     if name.endswith("_l0"):
        #         adding[name[:-1]] = param
        #     elif name.endswith("_l"): 
        #         adding[name+"0"] = param

        # for k, v in adding.items(): 
        #     state_dict[k] = v

        # iterate and assign
        for name, param in state_dict.items():
            if name not in own_state:
                in_dict_out_of_model.add(name)
                continue

            if isinstance(param, torch.nn.Parameter) or isinstance(param, torch.Tensor) :
                # backwards compatibility for serialized parameters
                in_model_out_of_dict.remove(name)
                param = param.data

            own_state[name].copy_(param)

        if verbose:
            print("in-model + out-of-dict ", "\n", in_model_out_of_dict, sep="")
            print("in-dict + out-of-model ", "\n", in_dict_out_of_model, sep="")



class RGN2_Refiner_Wrapper(torch.nn.Module):
    """ Wraps an engine for global refinement.

        Input example:  (uses https://github.com/hypnopump/en-transformer)
        refiner = RGN2_Refiner_Wrapper({
            'refiner_detach': true, # false
            'dim': 32,
            'depth': 4,
            'num_tokens': 21, # 1280+4, # 21 for int_seq
            'rel_pos_emb': true,
            'dim_head': 32,
            'heads': 2,
            'num_edge_tokens': None,
            'edge_dim': 8,
            'coors_hidden_dim': 16,
            'neighbors': 16,
            'num_adj_degrees': 1,
            'valid_neighbor_radius': 30,
            'checkpoint': None,
            # now it's about special args
            'refiner_detach': true,
            'feats_inputs': "int_seq", # "embedds"
        })
    """
    def __init__(self, **kwargs):
        super().__init__()
        # add args to class attrs
        for kw, arg in kwargs.items():
            self.__dict__[kw] = arg
        # add some args to defaults if not there
        if "refiner_detach" not in self.__dict__.keys():
            self.__dict__["refiner_detach"] = False

        # create dict of acceptable inputs
        self.refiner_args = {
            k:v for k,v in kwargs.items() \
            if k in set([
                "dim", "depth", "num_tokens", "rel_pos_emb", "dim_head",
                "heads", "num_edge_tokens", "edge_dim", "coors_hidden_dim",
                "neighbors", "only_sparse_neighbors", "num_adj_degrees",
                "adj_dim", "valid_neighbor_radius", "init_eps", "norm_rel_coors",
                "norm_coors_scale_init", "use_cross_product", "talking_heads",
                "checkpoint", "rotary_theta", "rel_dist_cutoff", "rel_dist_scale",
            ])
        }
        self.refiner = en_transformer.EnTransformer(**self.refiner_args)

    def forward(self, **data_dict):
        """ Corrects structure. """
        r_iters = []
        for i in range(max(1, data_dict["recycle"])):
            input_ = {
                k:v for k,v in data_dict.items()  \
                if k in set(["feats", "coors", "edges", "mask", "adj_mat"])
            }
            pred_feats, coors = self.refiner.forward(**input_)
            data_dict["coors"] = coors
            # data_dict["feats"] = pred_feats # commented for recycling

            if i != data_dict["recycle"]-1 and data_dict["inter_recycle"]:
                r_iters.append( coors.detach() )

        return pred_feats, coors, r_iters



