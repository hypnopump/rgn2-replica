# Author: Gurvinder Singh (@gurvindersingh)

import pytorch_lightning as pl
from datasets import load_from_disk
from .tokenizer import Tokenizer
from transformers import DataCollatorForLanguageModeling
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

class ProteinLMDataset():
    def __init__(self, ds, pfam_size, go_size, max_len=1024, columns=['input_ids','pfam_labels','go_labels']):
        self.max_len = max_len
        self.ds = ds
        self.ds.set_format(columns=columns)
        self.pmlb = MultiLabelBinarizer().fit([list(range(pfam_size))])
        self.gmlb = MultiLabelBinarizer().fit([list(range(go_size))])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = {}
        data['input_ids'] = torch.LongTensor(self.ds[idx]['input_ids'][:self.max_len])
        data['pfam_labels'] = torch.FloatTensor(self.pmlb.transform([self.ds[idx]['pfam_labels']])[0])
        data['go_labels'] = torch.FloatTensor(self.gmlb.transform([self.ds[idx]['go_labels']])[0])
        return data


class ProteinLMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ds_path,
        pfam_size,
        go_size,
        dtype="torch",
        bsize=16,
        num_procs=4,
        columns=["input_ids", "pfam_labels", "go_labels"],
        mlm_probability=0.15,
        pad_to_multiple_of=8,
        max_len=1024,
    ):
        super().__init__()
        (
            self.pfam_size,
            self.go_size,
            self.dtype,
            self.bsize,
            self.num_procs,
            self.max_len,
        ) = (pfam_size, go_size, dtype, bsize, num_procs, max_len)
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
                             self.pfam_size,
                             self.go_size,
                             max_len=self.max_len,
                             columns=self.columns),
            batch_size=self.bsize,
            collate_fn=self.data_collator,
            num_workers=self.num_procs,
        )

    def val_dataloader(self):
        return DataLoader(
            ProteinLMDataset(self.ds['valid'],
                             self.pfam_size,
                             self.go_size,
                             max_len=self.max_len,
                             columns=self.columns),
            batch_size=self.bsize,
            collate_fn=self.data_collator,
            num_workers=self.num_procs,
        )

