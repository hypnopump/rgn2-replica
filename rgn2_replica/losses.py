import torch
from torch import nn

class AminoBretLoss(nn.Module):

    def __init__(self, padding_token=-100, vocab_size=24):
        super().__init__()
        self.masked_loss = nn.CrossEntropyLoss(
            ignore_index=padding_token, reduction="none"
        )
        self.chunk_perm_loss = nn.CrossEntropyLoss(reduction="none")
        self.vocab_size = vocab_size

    def forward(
        self,
        logit_out: torch.Tensor,
        logit_chunk_perm: torch.Tensor,
        target: torch.Tensor,
        chunk_perm: torch.Tensor,
    ):
        """
        logit_out:  bs, len, vocab_size
        logit_chunk_perm: bs, 2 
        target: bs, len
        chunk_perm: (bs, 1)
        """
        global_petrub = 1 - chunk_perm

        masked_lm_loss = (
            self.masked_loss(logit_out.view(-1, self.vocab_size), target.view(-1))
            .view(target.shape[0], -1)
            .mean(1)
        )

        chunk_perm_loss = self.chunk_perm_loss(logit_chunk_perm, chunk_perm)
        loss = (chunk_perm * chunk_perm_loss) + (1 - global_petrub * masked_lm_loss)

        return loss.mean()
    
    
"""
vocab_size = 24
bs = 20
logit_out = torch.rand(bs, 10, vocab_size)
logit_chunk_perm = torch.rand(bs, 2)
target = torch.randint(1, 20, (bs, 10))
chunk_perm = torch.randint(0, 2, (bs,))

loss_func = AminoBretLoss(vocab_size=vocab_size)

loss = loss_func(logit_out, logit_chunk_perm, target, chunk_perm)
print(loss)

"""