# Author: Gurvinder Singh (@gurvindersingh)

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from x_transformers import Encoder, TransformerWrapper
from transformers.optimization import AdamW

from .tokenizer import Tokenizer
from .dataset import ProteinLMDataModule


class ProteinLMModel(pl.LightningModule):
    def __init__(
        self,
        vocabSz,
        lr,
        warmup_steps=2000,
        wd=0.0,
        bsize=4,
        gpus=1,
        epochs=4,
        max_len=1024,
        dim=512,
        depth=8,
        heads=8,
        ff_glu=True,
        use_rmsnorm=True,
        rotary_pos_emb=True,
        ff_dropout=0.0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = TransformerWrapper(
            num_tokens=vocabSz,
            max_seq_len=max_len,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                ff_glu=ff_glu,
                use_rmsnorm=use_rmsnorm,
                rotary_pos_emb=rotary_pos_emb,
                ff_dropout=ff_dropout,
            ),
        )

    def forward(self, x):
        return self.model(x).logits

    def process(self, batch):
        x, mask, labels = (
            batch["input_ids"],
            batch["attention_mask"].bool(),
            batch["labels"],
        )
        preds = self.model(x, mask=mask)
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(preds.view(-1, self.hparams.vocabSz), labels.view(-1))

    def training_step(self, batch, batch_idx):
        loss = self.process(batch)
        self.log_dict(
            {"train_loss": loss},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process(batch)
        self.log_dict(
            {"valid_loss": loss},
            on_step=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            trainLoader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (
                    len(trainLoader.dataset)
                    // (self.hparams.bsize * max(1, self.hparams.gpus))
                )
                // 1
                * float(self.hparams.epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.wd,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]