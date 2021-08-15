# Author: Gurvinder Singh (@gurvindersingh)

import argparse
import os
import pytorch_lightning as pl
from rgn2_replica.dataset import ProteinLMDataModule
from rgn2_replica.tokenizer import Tokenizer
from rgn2_replica.lmmodel import ProteinLMModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train protein language model')
    parser.add_argument("--ds", default="data/ur90_small", help="Dataset path")
    parser.add_argument("--accelerator", default="dp", help="Type of accelerator to train model. dp, ddp, ddp_spawn etc.")
    parser.add_argument("--output_dir", default="models_ckpt", help="Output directory to store model checkpoints")
    parser.add_argument("--gpus", default=0, help="Number of gpus to train on", type=int)
    parser.add_argument("--precision", default=32, help="Percision for training 16 or 32", type=int)
    parser.add_argument("--max_len", default=512, help="Maximum length per sequence", type=int)
    parser.add_argument("--epochs", default=10, help="Number of epochs", type=int)
    parser.add_argument("--bsize", default=16, help="Batch size per GPU", type=int)
    parser.add_argument("--lr", default=2e-4, help="Learning rate", type=float)
    
    return parser.parse_args()

def train(args):
    data = ProteinLMDataModule(args.ds, max_len=args.max_len)
    tokenizer = Tokenizer()
    model = ProteinLMModel(len(tokenizer.index), args.lr, max_len=args.max_len, bsize=args.bsize)
    os.makedirs(args.output_dir, exist_ok=True)
    pl.seed_everything(42)
    wandb_logger = WandbLogger(project="ProteinLM")
    checkpoint_callback = ModelCheckpoint(
                            monitor="valid_loss",
                            dirpath=args.output_dir,
                            filename="lmmodel-{epoch:02d}-{valid_loss:.2f}",
                            save_top_k=3,
                            mode="min",
                        )
    trainer = pl.Trainer(gpus=args.gpus, precision=args.precision,
                         accelerator=args.accelerator,max_epochs=args.epochs,
                         logger=wandb_logger,callbacks=[checkpoint_callback],
                        )
    trainer.fit(model, data)

if __name__ == '__main__':
    args = parse_arguments()
    train(args)
