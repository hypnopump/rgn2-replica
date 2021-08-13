from rgn2_replica.losses import AminoBretLoss
import torch

def test_aminobret_loss():
    vocab_size = 24
    bs = 20
    logit_out = torch.rand(bs, 10, vocab_size)
    logit_chunk_perm = torch.rand(bs, 2)
    target = torch.randint(1, 20, (bs, 10))
    chunk_perm = torch.randint(0, 2, (bs,))

    loss_func = AminoBretLoss(vocab_size=vocab_size)

    loss = loss_func(logit_out, logit_chunk_perm, target, chunk_perm)
    assert True

