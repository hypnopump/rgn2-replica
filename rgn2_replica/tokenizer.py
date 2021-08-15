# Author: Gurvinder Singh (@gurvindersingh)

from transformers import  PreTrainedTokenizer
class Tokenizer(PreTrainedTokenizer):
    # Taken from mp_nerf and extended based on ESM
    ALPHABETS = "ACDEFGHIKLMNPQRSTVWY_XBUZO"
    SPLTOKENS = ['<cls>','<eos>','<pad>','<mask>']

    def __init__(self, extra_tokens=[]):
        ext_tokens = self.SPLTOKENS + extra_tokens
        base_index = {aa:i for i,aa in enumerate(self.ALPHABETS)}
        ext_index = {k: i for k,i in zip(self.SPLTOKENS, list(range(len(self.ALPHABETS),len(self.ALPHABETS)+len(ext_tokens)))) }
        self.index = {**base_index, **ext_index}
        self.cls_token = '<mask>'
        self.mask_token = '<mask>'
        self._pad_token = '<mask>'
        self.padding_side = 'right'
        
    def __len__(self):
        return len(self.index)

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
        r['tokens'] = [self.convert_tokens_to_ids(self.cls_token)]+[self.convert_tokens_to_ids(list(r['sequence']))]
        return r
    
    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        return [1] + [0] * (len(token_ids)-1)
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.index[tokens]

        ids = []
        for token in tokens:
            ids.append(self.index[token])
        return ids
