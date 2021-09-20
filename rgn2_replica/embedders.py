import torch
from rgn2_replica.rgn2_utils import *
    

class ClaspEmbedder(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()

        from clasp import Transformer as ClaspTransformer, basic_aa_tokenizer
        self.tokenizer = basic_aa_tokenizer
        self.device = device

        # TODO: build encoders based on the config
        self.clasp_bioseq_encoder = ClaspTransformer(
            num_tokens = 23,
            dim = 768,
            depth = 12,
            seq_len = 512,
            sparse_attn = False,
            reversible=True
        )

        self.clasp_bioseq_encoder.load_state_dict(torch.load(config.embedder_checkpoint_path, map_location=device))
        self.clasp_bioseq_encoder.eval()

    def forward(self, aa_seq):
        """ Generates embeddings. 
            MP-NeRF: https://github.com/EleutherAI/mp_nerf
            Inputs: 
            * aa_seq: list of FASTA strs or 
                      torch.Tensor (B, L) according to MP-NeRF encoding
        """
        # format
        if isinstance(aa_seqs, torch.Tensor): 
            aa_seq = ids_to_embed_input(to_cpu(aa_seqs).tolist())

        with torch.no_grad():
            tokenized_seq = self.tokenizer(aa_seq, context_length=len(aa_seq), return_mask=False)
            all_embeddings = self.clasp_bioseq_encoder(to_device(tokenized_seq.unsqueeze(0), self.device), return_all_embeddings=True)

            # drop CLS embedding, return per-token embeddings only
            return all_embeddings[:, 1:]


class EsmEmbedder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        import esm
        self.embedder, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()
        self.device = device

    def forward(self, aa_seqs):
        """ Generates embeddings. 
            MP-NeRF: https://github.com/EleutherAI/mp_nerf
            Inputs: 
            * aa_seq: list of FASTA strs or 
                      torch.Tensor (B, L) according to MP-NeRF encoding
        """
        # format
        if isinstance(aa_seqs, torch.Tensor): 
            aa_seq = ids_to_embed_input(to_cpu(aa_seqs).tolist())
        # use ESM transformer
        REPR_LAYER_NUM = 33
        max_seq_len = max([len(aa_seq) for aa_seq in aa_seqs])

        batch_labels, batch_strs, batch_tokens = self.batch_converter(aa_seq)

        with torch.no_grad():
            results = self.embedder( 
                to_device(batch_tokens, self.device), 
                repr_layers=[REPR_LAYER_NUM], 
                return_contacts=False
            )
        # index 0 is for start token. so take from 1 one
        token_reps = results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :]
        return token_reps.detach()


def get_embedder(config, device):
    """Returns embedding model based on config.embedder_model

    Usage:
        config.embedder_model = 'clasp'
        OR
        config.embedder_model = 'esm1b'

        embedder = embedders.get_embedder(config, device)

        embeddings = embedder(aa_seq)
    """
    if config.embedder_model == 'clasp':
        print('Loading CLASP embedding model')

        config.emb_dim = 768
        config.embedder_checkpoint_path = '../clasp/data/run48_2021-07-18_13_31_19_step00005000.bioseq_encoder.pt'
        emb_model = ClaspEmbedder(config, device)
    else:
        print('Loading ESM-1b embedding model')

        config.emb_dim = 1280
        emb_model = EsmEmbedder(device)

    return emb_model.to(device)