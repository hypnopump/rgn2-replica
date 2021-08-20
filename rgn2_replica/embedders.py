import torch
    

class ClaspEmbedder(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()

        from clasp import CLASP, Transformer as ClaspTransformer, basic_aa_tokenizer
        self.tokenizer = basic_aa_tokenizer

        # TODO: build encoders based on the config
        self.clasp_model = CLASP(
            text_encoder = ClaspTransformer(
                num_tokens = 49408,
                dim = 768,
                depth = 12,
                seq_len = 1024,
                reversible=True
            ),
            bioseq_encoder = ClaspTransformer(
                num_tokens = 23,
                dim = 768,
                depth = 12,
                seq_len = 512,
                sparse_attn = False,
                reversible=True
            )
        )

        self.clasp_model.load_state_dict(torch.load(config.embedder_checkpoint_path, map_location=device))
        self.clasp_model.eval()

    def forward(self, aa_seq):
        with torch.no_grad():
            tokenized_seq = self.tokenizer(aa_seq, context_length=len(aa_seq), return_mask=False)
            all_embeddings = self.clasp_model.bioseq_encoder(tokenized_seq.unsqueeze(0), return_all_embeddings=True)

            # drop CLS embedding, return per-token embeddings only
            return all_embeddings[:, 1:]


class EsmEmbedder(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()

        import esm
        self.embedder, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()
        self.device = device

    def forward(self, aa_seq):
        # use ESM transformer
        REPR_LAYER_NUM = 33
        max_seq_len = len(aa_seq)

        _, _, batch_tokens = self.batch_converter([(None, aa_seq)])
        # batch_tokens = batch_tokens.unsqueeze(0)

        with torch.no_grad():
            results = self.embedder(batch_tokens.to(self.device), repr_layers=[REPR_LAYER_NUM], return_contacts=False)

        # index 0 is for start token. so take from 1 one
        return results["representations"][REPR_LAYER_NUM][..., 1:max_seq_len+1, :]


def get_embedder(config, device):
    """Returns embedding model based on config.embedder_model

    Usage:
        config.embedder_model = 'clasp'
        config.emb_dim = 768
        config.embedder_checkpoint_path = '../clasp/data/run48_2021-07-18_13_31_19_step00005000.pt'

        OR

        config.embedder_model = 'esm1b'
        config.emb_dim = 1280

        embedder = embedders.get_embedder(config, device)

        embeddings = embedder(aa_seq)
    """
    if config.embedder_model == 'clasp':
        print('Loading CLASP embedding model')
        emb_model = ClaspEmbedder(config, device)
    else:
        emb_model = EsmEmbedder(config, device)
        print('Loading ESM-1b embedding model')

    return emb_model.to(device)