# Author: Gurvinder Singh (@gurvindersingh)

import pytorch_lightning as pl
from datasets import load_from_disk
from .tokenizer import Tokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader

class ProteinLMDataset():
    def __init__(self, ds, max_len=1024, columns=['input_ids'], dtype='torch'):
        self.max_len = max_len
        self.ds = ds
        self.ds.set_format(dtype, columns=columns)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return {k:v[:self.max_len] for k,v in self.ds[idx].items()}
    

class ProteinLMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ds_path,
        dtype="torch",
        bsize=16,
        num_procs=4,
        columns=["input_ids"],
        mlm_probability=0.15,
        pad_to_multiple_of=8,
        max_len=1024,
    ):
        super().__init__()
        (
            self.dtype,
            self.bsize,
            self.num_procs,
            self.max_len,
        ) = (dtype, bsize, num_procs, max_len)
        self.ds = load_from_disk(ds_path)
        self.columns = columns
        tok = Tokenizer()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tok,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def train_dataloader(self):
        return DataLoader(
            ProteinLMDataset(self.ds['train'],
                             max_len=self.max_len,
                             columns=self.columns,
                             dtype=self.dtype),
            batch_size=self.bsize,
            collate_fn=self.data_collator,
            num_workers=self.num_procs,
        )

    def val_dataloader(self):
        return DataLoader(
            ProteinLMDataset(self.ds['valid'],
                             max_len=self.max_len,
                             columns=self.columns,
                             dtype=self.dtype),
            batch_size=self.bsize,
            collate_fn=self.data_collator,
            num_workers=self.num_procs,
        )

