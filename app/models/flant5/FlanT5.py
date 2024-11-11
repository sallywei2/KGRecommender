"""
This python file pulls out Flan-T5 trained model and related functions
from FlanT5_finetuning.ipynb for independent reference.
"""

import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

#import nltk
#nltk.download('punkt')
#from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

#"/content/drive/My Drive/Colab Notebooks/T5Ner"
# output_dir="" defaults to /content/lightning_logs/version_n/ where n is the run number (0, 1, 2, 3...)
OUTPUT_DIR = "lightning_logs" 

args_dict = dict(
    data_dir=None, # path for data files # unused for Text2KGBench
    output_dir=OUTPUT_DIR, # path to save the checkpoints
    default_root_dir=OUTPUT_DIR, # path to save the checkpoints
    model_name_or_path='google/flan-t5-small',
    tokenizer_name_or_path='google/flan-t5-small', #t5-small
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=4,
    num_dataloader_workers=1,
    val_check_interval=0.05, # run val/checkpoint after a fixed number of training batches. See https://lightning.ai/docs/pytorch/stable/common/trainer.html#pytorch_lightning.trainer.Trainer.params.val_check_interval
    # check_val_every_n_epoch = None # To deal with streaming data, set this to None and put an int > # training batches in val_check_interval
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False, # https://lightning.ai/docs/pytorch/stable/advanced/speed.html
    fp_16=False, 
    #fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    max_grad_norm=1,
    #max_grad_norm=0.5, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    seed=42,
)

args = argparse.Namespace(**args_dict)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparam):
        super(T5FineTuner, self).__init__()
        self.num_dataloader_workers = 6 # 6 CPU cores; original code used 2
        self.hparam = hparam

        self.model = T5ForConditionalGeneration.from_pretrained(
            hparam.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparam.model_name_or_path
        )
        self.save_hyperparameters()

        # manual optimization
        self.automatic_optimization = False
    
    def is_logger(self):
        return True

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    # Manual Optimization
    def training_step(self, batch, batch_idx):
        self.log("batch_idx", batch_idx)
        loss = self._step(batch) # compute loss

        self.manual_backward(loss) # manual optimization

        # manual optimization, replaces optimizer_step(...) below
        optimizer = self.optimizers()
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step() # learning rate scheduler

        self.log("train_loss",loss)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}


    # NotImplementedError: Support for `training_epoch_end` has been removed in v2.0.0.
    # `T5FineTuner` implements this method. You can use the `on_train_epoch_end` hook instead.
    # To access outputs, save them in-memory as instance attributes. You can find migration examples
    # in https://github.com/Lightning-AI/lightning/pull/16520.
    # def training_epoch_end(self, outputs):
    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.outputs).mean()
        #avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        self.log("avg_train_loss", avg_train_loss)
        tensorboard_logs = {"avg_train_loss": avg_train_loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.outputs = []
        return

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.outputs += loss.unsqueeze(0) # results
        self.log("step_val_loss", loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.outputs).mean()
        #avg_loss = torch.stack([x["val_loss"] for x in output_dict]).mean()
        
        self.log("val_loss",avg_loss)
        tensorboard_logs = {"val_loss": avg_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # "AdamW" was deprecated and suggested to use "torch.optim.AdamW" instead
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    # When performing automatic optimization:
    #    Error: The closure hasn't been executed. HINT: did you call `optimizer_closure()` in your `optimizer_step` hook?
    #    It could also happen because the `optimizer.step(optimizer_closure)` call did not execute it internally.
    # See also optimizer closures: https://lightning.ai/docs/pytorch/stable/common/optimization.html#use-closure-for-lbfgs-like-optimizers
    # 
    # Replaced by training_step(...) with manual optimization.
    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step() # learning rate scheduler

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparam)
        dataloader = DataLoader(train_dataset, batch_size=self.hparam.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=self.num_dataloader_workers)
        t_total = (
            (len(dataloader.dataset) //
             (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
            // self.hparam.gradient_accumulation_steps
            * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="validation", args=self.hparam)
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=self.num_dataloader_workers)

# usage: tokenizer = AutoTokenizer.from_pretrained("t5-small")
#        input_dataset = tokenize_dataset(tokenizer=tokenizer, dataset=dataset, type_path="train")
#
# type_path: torch.Dataset type_path parameter. "train", "test", "val"
def tokenize_dataset(tokenizer, dataset, type_path):
    custom_dataset = CustomDataset(tokenizer=tokenizer, dataset=dataset, type_path=type_path)
    if type_path == "train": # Only need to tokenize & pad training data
        # dunno what this is doing
        for i in range(len(custom_dataset)):
            _ = custom_dataset[i]
        tokenized_dataset = custom_dataset[0]
        print(tokenizer.decode(tokenized_dataset["source_ids"], skip_special_tokens=False))
        print(tokenizer.decode(tokenized_dataset["target_ids"], skip_special_tokens=False))
        return custom_dataset
    else:
        return custom_dataset

# to resume training in the middle if interrupted, or to load completed model from checkpoint
def load_model_from_checkpoint(CKPT_PATH, trainer=None):
    model = T5FineTuner.load_from_checkpoint(CKPT_PATH)

    checkpoint = torch.load(CKPT_PATH)

    if trainer:
        # restore from checkpoint/previous training progress
        # See: https://github.com/Lightning-AI/pytorch-lightning/issues/12274
        global_step_offset = checkpoint["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset

    # Fix for warning:
    #     You're resuming from a checkpoint that ended before the epoch ended and your dataloader is not resumable. 
    #     This can cause unreliable results if further training is done. Consider using an end-of-epoch checkpoint 
    #     or make your dataloader resumable by implementing the `state_dict` / `load_state_dict` interface.
    # Src: https://github.com/Lightning-AI/pytorch-lightning/issues/2798
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint) # Checkpoint contains only model state dict, it's not stored in a dict
        
    if 'lr_scheduler_state_dict' in checkpoint:
        model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print("Restored lr_scheduler_state_dict from checkpoint")
    if 'optimizer_state_dict' in checkpoint:
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Restored optimizer_state_dict from checkpoint")
    
    print("Loaded model from checkpoint:", CKPT_PATH)
    return model