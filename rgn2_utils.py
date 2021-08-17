import os
import torch
from sidechainnet.utils.sequence import ProteinVocabulary as VOCAB
VOCAB = VOCAB()

# data loading funcs copied from: 
# https://github.com/hypnopump/alphafold2/blob/main/alphafold2_pytorch/utils.py#L330

def ids_to_prottran_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for ids in x:
        chars = ' '.join([id2aa[i] for i in ids])
        chars = re.sub(r"[UZOB]", "X", chars)
        out.append(chars)

    return out

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
        return (None, ''.join(out))

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
    device = seq.device
    REPR_LAYER_NUM = 33
    max_seq_len = seq.shape[-1]
    embedd_inputs = ids_to_embed_input(seq.cpu().tolist())

    batch_labels, batch_strs, batch_tokens = batch_converter(embedd_inputs)
    with torch.no_grad():
        results = embedd_model(batch_tokens.to(device), repr_layers=[REPR_LAYER_NUM], return_contacts=False)
    # index 0 is for start token. so take from 1 one
    token_reps = results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :]
    return token_reps


def get_t5_embedd(seq, tokenizer, encoder, msa_data=None, device=None):
    """ Returns the ProtT5-XL-U50 embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * tokenizer:  tokenizer model: T5Tokenizer
        * encoder: encoder model: T5EncoderModel
                 ex: from transformers import T5EncoderModel, T5Tokenizer
                     model_name = "Rostlab/prot_t5_xl_uniref50"
                     tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
                     model = T5EncoderModel.from_pretrained(model_name)
                     # prepare model 
                     model = model.to(device)
                     model = model.eval()
                     if torch.cuda.is_available():
                         model = model.half()
        Outputs: tensor of (batch, n_seqs, L, embedd_dim)
            * n_seqs: number of sequences in the MSA. 1 for T5 models
            * embedd_dim: number of embedding dimensions. 1024 for T5 models
    """
    # get params and prepare
    device = seq.device if device is None else device
    embedd_inputs = ids_to_prottran_input(seq.cpu().tolist())
    
    # embedd - https://huggingface.co/Rostlab/prot_t5_xl_uniref50
    inputs_embedding = []
    shift_left, shift_right = 0, -1
    ids = tokenizer.batch_encode_plus(embedd_inputs, add_special_tokens=True,
                                                     padding=True, 
                                                     return_tensors="pt")
    with torch.no_grad():
        embedding = encoder(input_ids=torch.tensor(ids['input_ids']).to(device), 
                            attention_mask=torch.tensor(ids["attention_mask"]).to(device))
    # return (batch, seq_len, embedd_dim)
    token_reps = embedding.last_hidden_state[:, shift_left:shift_right]
    return token_reps.float()

