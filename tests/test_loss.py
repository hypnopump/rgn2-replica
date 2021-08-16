from rgn2_replica.losses import AminoBERTLoss
import torch

def test_aminobert_loss():
    vocab_size = 24
    bs = 20
    logit_out = torch.rand(bs, 10, vocab_size)
    logit_chunk_perm = torch.rand(bs, 2)
    target = torch.randint(1, 20, (bs, 10))
    chunk_perm = torch.randint(0, 2, (bs,))

    loss_func = AminoBERTLoss(vocab_size=vocab_size)

    loss = loss_func(logit_out, logit_chunk_perm, target, chunk_perm)
    assert True


def test_go_pfam_loss():
    num_classes = 45
    bs = 40
    target_pfam = torch.randint(0, num_classes, (bs,))
    target_go = torch.randint(0, num_classes, (bs,))
    logit_pfam = torch.rand(bs, num_classes)
    logit_go = torch.rand(bs, num_classes)
    weights = [2, 3]
    loss_func = GoPFAMLoss(weights=weights)
    loss = loss_func(
        logit_go=logit_go,
        logit_pfam=logit_pfam,
        target_go=target_go,
        target_pfam=target_pfam,
    )
    
    assert True
    
def test_loss_wrapper():
    num_classes = 45
    bs = 40
    target_pfam = torch.randint(0, num_classes, (bs,))
    target_go = torch.randint(0, num_classes, (bs,))
    logit_pfam = torch.rand(bs, num_classes)
    logit_go = torch.rand(bs, num_classes)
    weights = [2, 0.6]
    loss_go_pfam = GoPFAMLoss(weights=weights)


    vocab_size = 24
    logit_out = torch.rand(bs, 10, vocab_size)
    logit_chunk_perm = torch.rand(bs, 2)
    target = torch.randint(1, 20, (bs, 10))
    chunk_perm = torch.randint(0, 2, (bs,))
    loss_lm = AminoBERTLoss(vocab_size=vocab_size)


    combine_loss = LossWrapper(lm_loss=loss_lm, aux_loss=loss_go_pfam, weights=[2, 3])
    loss = combine_loss(
        logit_out=logit_out,
        logit_chunk_perm=logit_chunk_perm,
        logit_go=logit_go,
        logit_pfam=logit_pfam,
        target_go=target_go,
        target_pfam=target_pfam,
        chunk_perm=chunk_perm,
        target=target,
    )
    assert True
