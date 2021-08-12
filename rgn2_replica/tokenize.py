class Tokenizer():
    # Taken from mp_nerf and extended based on ESM
    ALPHABETS = "ACDEFGHIKLMNPQRSTVWY_XBUZO"
    SPLTOKENS = ['<cls>','<eos>','<pad>','<mask>']

    def __init__(self, extra_tokens=[]):
        ext_tokens = self.SPLTOKENS + extra_tokens
        base_index = {aa:i for i,aa in enumerate(self.ALPHABETS)}
        ext_index = {k: i for k,i in zip(self.SPLTOKENS, list(range(len(self.ALPHABETS),len(self.ALPHABETS)+len(ext_tokens)))) }
        self.index = {**base_index, **ext_index}
        self.cls_idx = self.index['<cls>']
        self.eos_idx = self.index['<eos>']
        self.pad_idx = self.index['<pad>']
        self.mask_idx = self.index['<mask>']

    def tokenize(self, r):
        """
        r: Dataset row with 'sequence' as one of the key
        Returns: Tokenized sequence

        To tokenize the dataset run the code as
        >>> from datasets load_from_disk
        >>> ds = load_from_disk('path_to_dataset')
        >>> tok = Tokenizer()
        >>> ds = ds.map(tok.tokenize, num_proc=4)
        >>> ds.save_to_disk('path_to_save_tokenized_ds')
        """
        r['tokens'] = [self.cls_idx]+[self.index[x] for x in r['sequence']]
        return r

