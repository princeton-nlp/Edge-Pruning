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
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
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
from transformers import AutoTokenizer
from modeling_fllama import FLlamaForCausalLM

import functools
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from accelerate import Accelerator

logger = logging.getLogger(__name__)

class FLlamaTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_node_sparsity = kwargs.pop('target_node_sparsity', 0.0)
        self.start_node_sparsity = kwargs.pop('start_node_sparsity', 0.0)
        
        self.edges_lr = kwargs.pop('edges_lr', 0.8)
        self.nodes_lr = kwargs.pop('nodes_lr', 0.8)
        self.reg_edges_lr = kwargs.pop('reg_edges_lr', 0.8)
        self.reg_nodes_lr = kwargs.pop('reg_nodes_lr', 0.8)
        self.warmup_steps = kwargs.pop('warmup_steps', 0)
        
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
        self.llama_model = kwargs.pop('llama_model', None)
        self.skip_node_loss_if_higher_sparsity = kwargs.pop('skip_node_loss_if_higher_sparsity', False)
        self.disable_node_loss = kwargs.pop('disable_node_loss', False)
        
        super().__init__(*args, **kwargs)
        
        self.llama_model.reset_all_log_alphas()
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.llama_model = self.accelerator.prepare(self.llama_model)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer, self.lr_scheduler = get_optimizers(
            self.model,
            self.edges_lr,
            self.nodes_lr,
            self.reg_edges_lr,
            self.reg_nodes_lr,
            num_training_steps,
            warmup_steps=self.warmup_steps
        )

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
        idxes = inputs.pop("idxes")
        _ = inputs.pop("labels")
        corr_input_ids = inputs.pop("corr_input_ids")
        input_ids = inputs.pop("input_ids")
        
        with torch.no_grad():
            # First get the logits from the llama model
            logits_llama = self.llama_model(input_ids=input_ids, **inputs).logits
            
            # Now run the corrupted inputs through it, and retain the activations
            corr_x = self.llama_model(input_ids=corr_input_ids, **inputs, output_writer_states=True).writer_states
        
        outputs = model(
            input_ids=input_ids,
            **inputs, 
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=None if self.disable_node_loss else self.get_current_node_target_sparsity(self.state.global_step),
            corr_x=corr_x,
        )
        
        edge_loss = outputs["edge_loss"]
        if self.disable_node_loss or (
            self.skip_node_loss_if_higher_sparsity and 
            outputs["model_node_sparsity"] > outputs["target_node_sparsity"]
        ):
            node_loss = 0
        else:
            node_loss = outputs["node_loss"]
        reg_loss = edge_loss + node_loss
        logits = outputs["logits"]
        
        kl_loss = 0
        for i in range(logits.shape[0]):
            if idxes[i] >= logits.shape[1]:
                continue
            logits_i = nn.functional.log_softmax(logits[i, idxes[i]], dim=-1)
            logits_llama_i = nn.functional.log_softmax(logits_llama[i, idxes[i]], dim=-1)
            
            kl_loss_component = nn.functional.kl_div(logits_i, logits_llama_i, reduction='sum', log_target=True)
            kl_loss = kl_loss + kl_loss_component
        kl_loss = kl_loss / logits.shape[0]
        
        loss = kl_loss + reg_loss
        outputs["loss"] = loss
        outputs["kl_loss"] = kl_loss

        # For logging purposes
        if 'target_node_sparsity' not in outputs:
            outputs['target_node_sparsity'] = -1
            outputs['model_node_sparsity'] = -1
        
        logger.info(f"@ {self.state.global_step} | Loss: {loss:.4f} | KL Loss: {kl_loss:.4f} | Edge Loss: {edge_loss:.4f} | Node Loss: {node_loss:.4f} | Model Edge Sparsity: {outputs['model_edge_sparsity']:.4f} | Model Node Sparsity: {outputs['model_node_sparsity']:.4f} | Target Edge Sparsity: {outputs['target_edge_sparsity']:.4f} | Target Node Sparsity: {outputs['target_node_sparsity']:.4f}")

        return (loss, outputs) if return_outputs else loss

@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(
        default="./data/datasets/merged/age/",
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
        default=64,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    start_edge_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial edge sparsity of the model."}
    )
    target_edge_sparsity: Optional[float] = field(
        default=1.2,
        metadata={"help": "The target edge sparsity of the model."}
    )
    start_node_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial node sparsity of the model."}
    )
    target_node_sparsity: Optional[float] = field(
        default=0.70,
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
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "The model to initialize from."},
    )
    ref_initialize_from: str = field(
        default=None,
        metadata={"help": "The model to initialize the reference model from."},
    )
    with_embedding_nodes: bool = field(
        default=False,
        metadata={"help": "Whether to allow pruning of embedding nodes."},
    )
    disable_linear_regularization_term: bool = field(
        default=False,
        metadata={"help": "Whether to disable the linear regularization term."},
    )
    disable_node_loss: bool = field(
        default=False,
        metadata={"help": "Whether to disable the node loss."},
    )

def load_datasets(dataset_path, max_train_samples, max_eval_samples):
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    
    if max_train_samples is not None and max_train_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train_samples))
    if "validation" not in dataset: # Not a big deal, we don't do eval during training anyway
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["train"]
        })
    if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset

def format_instruction(entry):
    ip = entry["input"]
    ip = ip[:ip.rfind(" is")]
    ip = f"[INST] <<SYS>>\nEvaluate the following boolean expression as either 'True' or 'False'.\n<</SYS>>\n\n" + \
        f"{ip} [/INST] '"
    corr_ip = entry["corr_input"]
    corr_ip = corr_ip[:corr_ip.rfind(" is")]
    corr_ip = f"[INST] <<SYS>>\nEvaluate the following boolean expression as either 'True' or 'False'.\n<</SYS>>\n\n" + \
        f"{corr_ip} [/INST] '"
    return ip, corr_ip, entry["target"].strip()

class DataCollatorBool:
    def __init__(
        self, 
        tokenizer,
        max_length
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        input_ids_all = []
        corr_input_ids_all = []
        labels_all = []
        idxes = []
        
        for example in examples:
            text, corr_text, target = format_instruction(example)
            
            idx = len(self.tokenizer.encode(text)) - 1
            input_ids = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            ).input_ids[0]
            corr_input_ids = self.tokenizer(
                corr_text, 
                return_tensors="pt",
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            ).input_ids[0]
            labels = self.tokenizer.convert_tokens_to_ids(target)
            if type(labels) == list:
                labels = labels[0]
            
            input_ids_all.append(input_ids)
            corr_input_ids_all.append(corr_input_ids)
            labels_all.append(labels)
            idxes.append(idx)
        
        batch = {
            "input_ids": torch.stack(input_ids_all),
            "corr_input_ids": torch.stack(corr_input_ids_all),
            "labels": torch.LongTensor(labels_all),
            "idxes": torch.LongTensor(idxes),
        }

        return batch     
    
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
        if 'read_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'write_log_alpha' in n:
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
    
    model = FLlamaForCausalLM.from_pretrained(
        model_args.initialize_from, 
        with_embedding_nodes=model_args.with_embedding_nodes, 
        disable_linear_regularization_term=model_args.disable_linear_regularization_term
    )
    llama_model_initialize_from = model_args.ref_initialize_from if model_args.ref_initialize_from is not None else model_args.initialize_from
    llama_model = FLlamaForCausalLM.from_pretrained(
        llama_model_initialize_from,
        with_embedding_nodes=model_args.with_embedding_nodes,
        disable_linear_regularization_term=model_args.disable_linear_regularization_term
    )
    
    model.reset_all_log_alphas()
    llama_model.reset_all_log_alphas()
    
    tokenizer = AutoTokenizer.from_pretrained(llama_model_initialize_from)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # Preserve position information
    
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
    collator = DataCollatorBool(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )

    # Initialize our Trainer
    trainer = FLlamaTrainer(
        model=model,
        llama_model=llama_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_node_sparsity=data_args.start_node_sparsity,
        target_node_sparsity=data_args.target_node_sparsity,
        skip_node_loss_if_higher_sparsity=data_args.stop_optimizing_node_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
        edges_lr=data_args.edge_learning_rate,
        nodes_lr=data_args.node_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_nodes_lr=data_args.reg_node_learning_rate,
        warmup_steps=training_args.warmup_steps,
        disable_node_loss=model_args.disable_node_loss,
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

    kwargs = {"finetuned_from": "codellama"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()