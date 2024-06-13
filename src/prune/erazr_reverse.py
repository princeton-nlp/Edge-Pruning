#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import torch
import pickle
import random
import math
import sys
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

import transformers
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from torch.optim import AdamW

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)
from modeling_erazr import ErazrModelForSequenceTransformation, ErazrTokenizer, ErazrConfig, get_config_weights_and_vocab

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

class ErazrTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_node_sparsity = kwargs.pop('target_node_sparsity', 0.0)
        self.start_node_sparsity = kwargs.pop('start_node_sparsity', 0.0)
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_node_sparsity_warmup_steps" in kwargs:
            self.num_node_sparsity_warmup_steps = kwargs.pop('num_node_sparsity_warmup_steps')
        else:
            self.num_node_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', self.num_edge_sparsity_warmup_steps)
        _ = kwargs.pop('num_sparsity_warmup_steps', None)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.tracr_model = kwargs.pop('tracr_model', None)
        self.skip_node_loss_if_higher_sparsity = kwargs.pop('skip_node_loss_if_higher_sparsity', False)
        self.zero_ablation = kwargs.pop('zero_ablation', False)
        self.disable_node_loss = kwargs.pop('disable_node_loss', False)
        super().__init__(*args, **kwargs)

    def get_current_edge_target_sparsity(self, global_step):
        if global_step < self.num_edge_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_edge_sparsity + (self.target_edge_sparsity - self.start_edge_sparsity) * 
                    global_step / self.num_edge_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_edge_sparsity) + (math.log(1 - self.target_edge_sparsity) - 
                    math.log(1 - self.start_edge_sparsity)) * global_step / self.num_edge_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_edge_sparsity
        
    def get_current_node_target_sparsity(self, global_step):
        if global_step < self.num_node_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_node_sparsity + (self.target_node_sparsity - self.start_node_sparsity) * 
                    global_step / self.num_node_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_node_sparsity) + (math.log(1 - self.target_node_sparsity) - 
                    math.log(1 - self.start_node_sparsity)) * global_step / self.num_node_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_node_sparsity

    def compute_loss(self, model, inputs, return_outputs=False):
        _ = inputs.pop("labels")
        corr_input_ids = inputs.pop("corr_input_ids")
        input_ids = inputs.pop("input_ids")
        
        with torch.no_grad():
            # First get the logits from the Tracr (still bundled up as Erazr) model
            logits_tracr = self.tracr_model(input_ids=input_ids, **inputs, return_dict=False)[0]
            # Now run the corrupted inputs through it, and retain the activations
            if self.zero_ablation:
                corr_x = None
            else:
                corr_x = self.tracr_model(input_ids=corr_input_ids, **inputs, output_writer_states=True, return_dict=True).writer_states
        
        outputs = model(
            input_ids=input_ids,
            **inputs, 
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=self.get_current_node_target_sparsity(self.state.global_step),
            corr_x=corr_x,
        )
        
        edge_loss = outputs[6]
        if self.disable_node_loss or (
            self.skip_node_loss_if_higher_sparsity and 
            outputs["model_node_sparsity"] > outputs["target_node_sparsity"]
        ):
            node_loss = 0
        else:
            node_loss = outputs[7]
        reg_loss = edge_loss + node_loss
        logits = outputs[0]
        
        # The KL loss does not apply to idx 0 (it is the BOS token)
        kl_loss = nn.functional.kl_div(
            nn.functional.log_softmax(logits[:, 1:, :], dim=-1),
            nn.functional.log_softmax(logits_tracr[:, 1:, :], dim=-1),
            reduction='batchmean',
            log_target=True
        )
        
        loss = kl_loss + reg_loss
        # HF trainer is awful and somehow loses the first copy of logits plus we should remove the last hidden states
        outputs = (outputs[0], outputs[0]) + (outputs[2:]) + (kl_loss, loss,)

        return (loss, outputs) if return_outputs else loss

@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(
        default="./data/datasets/reverse-t3-s3",
        metadata={"help": "The path to the directory with the JSON files of the task."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=4,
        metadata={"help": "The input sequence length after tokenization including BOS."}
    )
    seq_length: Optional[int] = field(
        default=4,
        metadata={"help": "The input sequence length after tokenization including BOS."}
    )
    start_edge_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial edge sparsity of the model."}
    )
    target_edge_sparsity: Optional[float] = field(
        default=0.92,
        metadata={"help": "The target edge sparsity of the model."}
    )
    start_node_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial node sparsity of the model."}
    )
    target_node_sparsity: Optional[float] = field(
        default=0.16,
        metadata={"help": "The target node sparsity of the model."}
    )
    stop_optimizing_node_if_higher_sparsity: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to stop optimizing the node sparsity if it is higher than the target."}
    )
    num_sparsity_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "The number of steps to reach the target sparsity."}
    )
    edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    node_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_node_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    warmup_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The type of warmup to use for the regularization term."}
    )
    zero_ablation: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to zero out the ablated tokens."}
    )
    disable_node_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the node loss."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    initialize_from: str = field(
        default="models/reverse.tracr.pkl",
        metadata={"help": "The model to initialize from."},
    )

def format_instance(instance, split):
    if isinstance(instance, dict) and "min_steps" in instance:
        return {
            "tokens": instance["tokens"],
            "split": split,
            "min_steps": instance["min_steps"],
        }
    else:
        return {
            "tokens": instance,
            "split": split,
        }

def load_datasets(dataset_path, max_train_samples, max_eval_samples):
    dataset_ = load_from_disk(dataset_path)
    dataset = DatasetDict({
        "train": dataset_,
        "validation": dataset_,
    })
    if max_train_samples is not None and max_train_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train_samples))
    if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset

def unstringify(l):
    for i in range(1, len(l)):
        l[i] = int(l[i])
    return l

class DataCollatorReverse:
    def __init__(
        self, 
        tokenizer,
        length,
    ):
        self.tokenizer = tokenizer
        self.length = length

    def __call__(self, examples):
        input_ids_all = []
        corr_input_ids_all = []
        labels_all = []
        
        for example in examples:
            seq = unstringify(example["seq"])
            target = unstringify(example["target"])
            corr_seq = unstringify(example["corr_seq"])
            
            input_ids = self.tokenizer([seq], return_tensors="pt")[0]
            corr_input_ids = self.tokenizer([corr_seq], return_tensors="pt")[0]
            labels = self.tokenizer([target], return_tensors="pt")[0]
            
            assert input_ids.shape[0] == self.length, f"Input length is {input_ids.shape[0]}, expected {self.length}"
            assert corr_input_ids.shape[0] == self.length, f"Corrupted length is {corr_input_ids.shape[0]}, expected {self.length}"
            assert labels.shape[0] == self.length, f"Target length is {labels.shape[0]}, expected {self.length}"
            
            input_ids_all.append(input_ids)
            corr_input_ids_all.append(corr_input_ids)
            labels_all.append(labels)
        
        return {
            "input_ids": torch.stack(input_ids_all),
            "corr_input_ids": torch.stack(corr_input_ids_all),
            "labels": torch.stack(labels_all),
        }        

def eval_fn(eval_pred): 
    logits, target_edge_sparsity, target_node_sparsity, model_edge_sparsity, model_node_sparsity, edge_loss, node_loss, kl_loss, loss = eval_pred.predictions
    if len(model_edge_sparsity.shape) > 0:
        model_edge_sparsity = model_edge_sparsity[0].item()
        model_node_sparsity = model_node_sparsity[0].item()
        target_edge_sparsity = target_edge_sparsity[0].item()
        target_node_sparsity = target_node_sparsity[0].item()
    else:
        model_edge_sparsity = model_edge_sparsity.item()
        model_node_sparsity = model_node_sparsity.item()
        target_edge_sparsity = target_edge_sparsity.item()
        target_node_sparsity = target_node_sparsity.item()
    
    predictions = np.argmax(logits[:, 1:, :], axis=-1)
    labels = eval_pred.label_ids[:, 1:]

    n = 0
    accuracy = 0
    for i in range(predictions.shape[0]):
        n += 1
        if (predictions[i] == labels[i]).all():
            accuracy += 1
    accuracy /= n
    
    correct = (predictions == labels).all(axis=1)
    accuracy = correct.sum().item() / correct.shape[0]
    
    kl_loss = kl_loss.mean().item()
    edge_loss = edge_loss.mean().item()
    node_loss = node_loss.mean().item()
    
    return {
        "eval_accuracy": accuracy,
        "model_edge_sparsity": model_edge_sparsity,
        "model_node_sparsity": model_node_sparsity,
        "target_edge_sparsity": target_edge_sparsity,
        "target_node_sparsity": target_node_sparsity,
        "kl_loss": kl_loss,
        "edge_loss": edge_loss,
        "node_loss": node_loss,
    }
    
def freeze_all_except_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def get_optimizers(model, edges_lr, nodes_lr, reg_edges_lr, reg_nodes_lr, num_training_steps, warmup_steps=0):
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []

    for n, p in model.named_parameters():
        if 'entitywise_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda_edge' in n:
            optimizer_2_group.append(p)
        elif 'sparsity_lambda_node' in n:
            optimizer_4_group.append(p)
    
    optimizer = AdamW(
        [
            {
                'params': optimizer_1_group,
                'lr': edges_lr,
            },
            {
                'params': optimizer_2_group,
                'maximize': True,
                'lr': reg_edges_lr,
            },
            {
                'params': optimizer_3_group,
                'lr': nodes_lr,
            },
            {
                'params': optimizer_4_group,
                'maximize': True,
                'lr': reg_nodes_lr,
            } 
        ],
        lr=edges_lr
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_datasets(data_args.dataset_path, data_args.max_train_samples, data_args.max_eval_samples)
    n_train = len(raw_datasets["train"])
    
    config, tok_embed, pos_embed, blocks_embeds, vocab, bos, pad, unembedding_mtx = get_config_weights_and_vocab(model_args.initialize_from)
    model = ErazrModelForSequenceTransformation(config).to("cuda")
    model.load_everything(tok_embed, pos_embed, unembedding_mtx, blocks_embeds)
    model.set_edge_threshold_for_deterministic(0.005)
    model.set_node_threshold_for_deterministic(0.005)
    
    tracr_model = ErazrModelForSequenceTransformation(config).to("cuda")
    tracr_model.load_everything(tok_embed, pos_embed, unembedding_mtx, blocks_embeds)
    
    tokenizer = ErazrTokenizer(vocab, bos, pad)
    
    freeze_all_except_pruning_params(model)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        # We don't have a validation dataset, so we'll just use the test dataset.
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    # Data collator
    collator = DataCollatorReverse(
        tokenizer=tokenizer,
        length=data_args.seq_length
    )
    
    optimizers = get_optimizers(
        model, 
        edges_lr=data_args.edge_learning_rate,
        nodes_lr=data_args.node_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_nodes_lr=data_args.reg_node_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps,
    )

    # Initialize our Trainer
    trainer = ErazrTrainer(
        model=model,
        tracr_model=tracr_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_node_sparsity=data_args.start_node_sparsity,
        target_node_sparsity=data_args.target_node_sparsity,
        skip_node_loss_if_higher_sparsity=data_args.stop_optimizing_node_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
        zero_ablation=data_args.zero_ablation,
        disable_node_loss=data_args.disable_node_loss,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
        )
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Run final evaluation
    model.eval()
    n = 0
    accuracy = 0
    for i in range(0, len(eval_dataset), training_args.eval_batch_size):
        input_ids = []
        corr_input_ids = []
        labels = []
        
        for j in range(i, min(i + training_args.eval_batch_size, len(eval_dataset))):
            input_ids_ = tokenizer([unstringify(eval_dataset[j]["seq"])], return_tensors="pt")[0].to("cuda")
            corr_input_ids_ = tokenizer([unstringify(eval_dataset[j]["corr_seq"])], return_tensors="pt")[0].to("cuda")
            labels_ = tokenizer([unstringify(eval_dataset[j]["target"])], return_tensors="pt")[0].to("cuda")
            
            input_ids.append(input_ids_)
            corr_input_ids.append(corr_input_ids_)
            labels.append(labels_)
            
        input_ids = torch.stack(input_ids)
        corr_input_ids = torch.stack(corr_input_ids)
        labels = torch.stack(labels)
        
        with torch.no_grad():
            if data_args.zero_ablation:
                corr_x = None
            else:
                corr_x = tracr_model(corr_input_ids, output_writer_states=True, return_dict=True).writer_states
            outputs = model(input_ids, corr_x=corr_x, return_dict=True)
            preds = outputs.logits.argmax(-1)
            
            for j in range(len(input_ids)):
                n += 1
                if torch.all(preds[j, 1:] == labels[j, 1:]):
                    accuracy += 1
    print("Accuracy:", accuracy / n)
    print("Edges:")
    print(model.get_edges())

    kwargs = {"finetuned_from": "tracr"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()