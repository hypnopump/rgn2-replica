import re
import torch
from sidechainnet.utils.sequence import ProteinVocabulary as VOCAB

# random hacks - device utils for pyTorch - saves transfers
to_cpu = lambda x: x.cpu() if x.is_cuda else x
to_device = lambda x, device: x.to(device) if x.device != device else x

VOCAB = VOCAB()

# data loading funcs copied from: 
# https://github.com/hypnopump/alphafold2/blob/main/alphafold2_pytorch/utils.py#L330

def ids_to_embed_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(ids_to_embed_input(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')

    if all(map(lambda c: isinstance(c, str), out)):
        return (None, ''.join(out).replace("_", ""))

    return out


def get_esm_embedd(seq, embedd_model, batch_converter, msa_data=None):
    """ Returns the ESM embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embedd_model: ESM model (see train_end2end.py for an example)
        * batch_converter: ESM batch converter (see train_end2end.py for an example)
        Outputs: tensor of (b, L, embedd_dim)
            * embedd_dim: number of embedding dimensions. 1280 for ESM-1b
    """
    # use ESM transformer
    device = next(embedd_model.parameters()).device
    REPR_LAYER_NUM = 33
    max_seq_len = seq.shape[-1]
    embedd_inputs = ids_to_embed_input( to_cpu(seq).tolist() )

    batch_labels, batch_strs, batch_tokens = batch_converter(embedd_inputs)
    with torch.no_grad():
        results = embedd_model( to_device(batch_tokens, device), repr_layers=[REPR_LAYER_NUM], return_contacts=False )
    # index 0 is for start token. so take from 1 one
    token_reps = results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :]
    return token_reps.detach()


def seqs_from_fasta(fasta_file, names=False): 
    """ Reads protein sequences from FASTA files. """
    seqs, names = [], []
    with open(fasta_file, "r") as f: 
        lines = f.readlines()
        for i, line in enumerate(lines): 
            if line[0] not in {">", ";"}: 
                names.append( lines[i-1][1:].replace(" ", "_").replace("\n", "") )
                seqs.append( line.replace("\n", "") )

    seqs = [re.sub(r'[^a-zA-Z]','', seq).upper() for seq in seqs]
    return seqs if not names else (seqs, names)


