import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F


class AminoBERTLoss(nn.Module):
    def __init__(self, padding_token=-100, vocab_size=24):
        super().__init__()
        self.masked_loss = nn.CrossEntropyLoss(
            ignore_index=padding_token
        )
        self.chunk_perm_loss = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size

    def forward(
        self, logit_out=None, logit_chunk_perm=None, target=None, chunk_perm=None,
    ):
        """
        logit_out:  (bs, len, vocab_size) tensor
        logit_chunk_perm: (bs, 2) tensor
        target: (bs, len) tensor
        chunk_perm: (bs, 1)
        """

        # to do: Check Logic
        global_petrub = 1 - chunk_perm

        masked_lm_loss = self.masked_loss(logit_out.view(-1, self.vocab_size), target.view(-1))

        chunk_perm_loss = self.chunk_perm_loss(logit_chunk_perm, chunk_perm)
        loss = (chunk_perm * chunk_perm_loss) + (1 - global_petrub * masked_lm_loss)

        return loss


class GoPFAMLoss(nn.Module):
    def __init__(self, weights=[1., 1.]):
        """
        weights: [float, float] weights for combining loss for go and pfam
        """
        super().__init__()
        self.weights = weights
        self.go_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.pfam_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logit_go=None, logit_pfam=None, target_go=None, target_pfam=None):
        """
        logit_go: (bs, go_n_classes)
        logit_pfam: (bs, pfam_n_classes)
        target_go: (bs, go_n_classes)
        target_pfam: (bs, pfam_n_classes)
        """

        # When label belongs to class 0 means no label, we ignore the loss
        go_weights = torch.argmax(target_go, dim=-1).clamp(0,1)
        pfam_weights = torch.argmax(target_pfam, dim=-1).clamp(0,1)

        go_loss = (self.go_loss(logit_go, target_go).mean(dim=-1) * go_weights).sum()
        pfam_loss = (self.pfam_loss(logit_pfam, target_pfam).mean(dim=-1) * pfam_weights).sum()

        if go_weights.sum() > 0:
             go_loss = go_loss/go_weights.sum()
        if pfam_weights.sum() > 0:
             pfam_loss = pfam_loss/pfam_weights.sum()

        combined_loss = go_loss * self.weights[0] + pfam_loss * self.weights[1]
        return combined_loss


class LossWrapper(nn.Module):
    def __init__(self, lm_loss: None, aux_loss: None, weights=[1., 1.]):
        """
        Combines AminoBERTLoss with GoPFAMLoss
        """
        super().__init__()
        self.lm_loss = lm_loss
        self.aux_loss = aux_loss
        self.weights = weights

    def forward(
        self,
        logit_out=None,
        logit_chunk_perm=None,
        logit_go=None,
        logit_pfam=None,
        target_go=None,
        target_pfam=None,
        chunk_perm=None,
        target=None,
    ):

        lm_loss = self.lm_loss(
            logit_out=logit_out,
            logit_chunk_perm=logit_chunk_perm,
            target=target,
            chunk_perm=chunk_perm,
        )
        aux_loss = self.aux_loss(
            logit_go=logit_go,
            logit_pfam=logit_pfam,
            target_go=target_go,
            target_pfam=target_pfam,
        )
        combined_loss = lm_loss * self.weights[0] + aux_loss * self.weights[1]

        return combined_loss