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
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
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
)   # Very hacky but the imports are annoying otherwise
from modeling_fpt2 import FPT2LMHeadModel

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

class FPT2InfoTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_layer_sparsity = kwargs.pop('target_layer_sparsity', 0.0)
        self.start_layer_sparsity = kwargs.pop('start_layer_sparsity', 0.0)
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_layer_sparsity_warmup_steps" in kwargs:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_layer_sparsity_warmup_steps')
        else:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', self.num_edge_sparsity_warmup_steps)
        _ = kwargs.pop('num_sparsity_warmup_steps', None)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.gpt2_model = kwargs.pop('gpt2_model', None)
        self.skip_layer_loss_if_higher_sparsity = kwargs.pop('skip_layer_loss_if_higher_sparsity', False)
        self.device_count = torch.cuda.device_count()
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
        
    def get_current_layer_target_sparsity(self, global_step):
        if global_step < self.num_layer_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_layer_sparsity + (self.target_layer_sparsity - self.start_layer_sparsity) * 
                    global_step / self.num_layer_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_layer_sparsity) + (math.log(1 - self.target_layer_sparsity) - 
                    math.log(1 - self.start_layer_sparsity)) * global_step / self.num_layer_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_layer_sparsity

    def compute_loss(self, model, inputs, return_outputs=False):
        start_idxes = inputs.pop("start_idxes")
        end_idxes = inputs.pop("end_idxes")
        _ = inputs.pop("labels")
        corr_input_ids = inputs.pop("corr_input_ids")
        input_ids = inputs.pop("input_ids")
        
        bsz = input_ids.shape[0]
        
        with torch.no_grad():
            # First get the logits from the GPT-2 model
            logits_gpt2 = self.gpt2_model(input_ids=input_ids, **inputs).logits
            
            # Now run the corrupted inputs through it, and retain the activations
            corr_x = self.gpt2_model(
                input_ids=corr_input_ids, 
                **inputs,
                output_writer_states=True
            ).writer_states

            # Reshape corr_x in case we have distributed training
            tgt_shape = (-1, bsz // self.device_count, *corr_x.shape[2:])
            corr_x = corr_x.reshape(tgt_shape)
        
        outputs = model(
            input_ids=input_ids,
            **inputs, 
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=self.get_current_layer_target_sparsity(self.state.global_step),
            corr_x=corr_x
        )
        
        reg_edge_loss = outputs["edge_loss"]
        if self.skip_layer_loss_if_higher_sparsity and outputs["model_node_sparsity"] > outputs["target_node_sparsity"]:
            reg_layer_loss = 0
        else:
            reg_layer_loss = outputs["node_loss"]
        reg_loss = reg_edge_loss + reg_layer_loss
        logits = outputs["logits"]
        
        kl_loss = 0
        for i in range(logits.shape[0]):
            logits_i = nn.functional.log_softmax(logits[i, start_idxes[i]:end_idxes[i]], dim=-1)
            logits_gpt2_i = nn.functional.log_softmax(logits_gpt2[i, start_idxes[i]:end_idxes[i]], dim=-1)
            
            kl_loss_component = nn.functional.kl_div(logits_i, logits_gpt2_i, reduction='batchmean', log_target=True)
            kl_loss += kl_loss_component
        kl_loss /= logits.shape[0]
        
        loss = kl_loss + reg_loss
        outputs["loss"] = loss
        outputs["kl_loss"] = kl_loss

        return (loss, outputs) if return_outputs else loss

@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(
        default="./data/datasets/example_custom.jsonl",
        metadata={"help": "The path to the directory with the JSON files of the task."},
    )
    train_split: Optional[str] = field(
        default="train",
        metadata={"help": "The split to use for training."},
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
        default=0.97,
        metadata={"help": "The target edge sparsity of the model."}
    )
    start_layer_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial layer sparsity of the model."}
    )
    target_layer_sparsity: Optional[float] = field(
        default=0.72,
        metadata={"help": "The target layer sparsity of the model."}
    )
    stop_optimizing_layer_if_higher_sparsity: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to stop optimizing the layer sparsity if it is higher than the target."}
    )
    num_sparsity_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "The number of steps to reach the target sparsity."}
    )
    edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    warmup_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The type of warmup to use for the regularization term."}
    )
    with_embedding_nodes: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to include the embedding nodes"}
    )
    disable_linear_reg_term: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the linear regularization term."}
    )
    disable_node_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable node loss."}
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
        default="gpt2",
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

def load_datasets(dataset_path, max_train_samples, max_eval_samples, train_split="train"):
    if os.path.exists(dataset_path):
        if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_path)["train"]
            if "split" in dataset:
                # Filter into train and eval
                dataset = DatasetDict({
                    "train": dataset.filter(lambda x: x["split"] == "train"),
                    "validation": dataset.filter(lambda x: x["split"] == "validation"),
                })
            else:
                logger.warning("No split information found in the dataset. Using the whole dataset for both train and eval.")
                # Use the whole dataset for both train and eval
                dataset = DatasetDict({
                    "train": dataset,
                    "validation": dataset,
                })
        else:
            dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    if "validation" not in dataset:
        assert max_eval_samples is not None, "Validation set is missing! (val)"
        assert max_train_samples is not None, "Validation set is missing! (train)"
        dataset = DatasetDict({
            train_split: dataset[train_split].select(range(max_train_samples)),
            "validation": dataset[train_split].select(range(max_train_samples, max_train_samples+max_eval_samples)),
        })
    else:
        if max_train_samples is not None and max_train_samples < len(dataset[train_split]):
            dataset[train_split] = dataset[train_split].select(range(max_train_samples))
        if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
            dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset

class DataCollatorCustom:
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
        start_idxes = []
        end_idxes = []
        
        for example in examples:
            text = example["clean"]
            corr_text = example["corrupted"]
            
            assert "<predict>" in text and "</predict>" in text, f"Text does not contain <predict> and </predict>: {text}"
            non_loss_portion = text[:text.find("<predict>")]
            loss_portion = text[text.find("<predict>")+len("<predict>"):text.find("</predict>")]
            final_portion = text[text.find("</predict>")+len("</predict>"):]
            if "<predict>" in corr_text:
                corr_text = corr_text[:corr_text.find("<predict>")] + corr_text[corr_text.find("</predict>")+len("</predict>"):]
            
            len_non_loss = self.tokenizer(non_loss_portion.strip(), return_tensors="pt").input_ids.shape[1]
            len_up_to_loss = self.tokenizer((non_loss_portion + loss_portion).strip(), return_tensors="pt").input_ids.shape[1]
            input_ids = self.tokenizer(
                non_loss_portion + loss_portion + final_portion, 
                return_tensors="pt", 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            ).input_ids[0]
            corr_input_ids = self.tokenizer(corr_text, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True).input_ids[0]
            labels = input_ids.clone()
            labels[:len_non_loss] = -100
            labels[len_up_to_loss:] = -100

            input_ids_all.append(input_ids)
            corr_input_ids_all.append(corr_input_ids)
            labels_all.append(labels)
            
            start_idxes.append(len_non_loss-1)
            end_idxes.append(len_up_to_loss-1)
        
        batch = {
            "input_ids": torch.stack(input_ids_all),
            "corr_input_ids": torch.stack(corr_input_ids_all),
            "labels": torch.stack(labels_all),
            "start_idxes": torch.LongTensor(start_idxes),
            "end_idxes": torch.LongTensor(end_idxes),
        }

        return batch     

def eval_fn(eval_pred): 
    logits, target_edge_sparsity, target_layer_sparsity, model_edge_sparsity, model_layer_sparsity, reg_edge_loss, reg_layer_loss, kl_loss = eval_pred.predictions
    if len(model_edge_sparsity.shape) > 0:
        model_edge_sparsity = model_edge_sparsity[0].item()
        model_layer_sparsity = model_layer_sparsity[0].item()
        target_edge_sparsity = target_edge_sparsity[0].item()
        target_layer_sparsity = target_layer_sparsity[0].item()
    else:
        model_edge_sparsity = model_edge_sparsity.item()
        model_layer_sparsity = model_layer_sparsity.item()
        target_edge_sparsity = target_edge_sparsity.item()
        target_layer_sparsity = target_layer_sparsity.item()
    
    predictions = np.argmax(logits, axis=-1)[:, :-1]
    labels = eval_pred.label_ids[:, 1:]

    eval_mask = (labels != -100).astype(int)
    predictions = predictions * eval_mask
    labels = labels * eval_mask
    
    correct = (predictions == labels).all(axis=1)
    accuracy = correct.sum().item() / correct.shape[0]
    
    kl_loss = kl_loss.mean().item()
    reg_edge_loss = reg_edge_loss.mean().item()
    reg_layer_loss = reg_layer_loss.mean().item()
    
    return {
        "eval_accuracy": accuracy,
        "model_edge_sparsity": model_edge_sparsity,
        "model_layer_sparsity": model_layer_sparsity,
        "target_edge_sparsity": target_edge_sparsity,
        "target_layer_sparsity": target_layer_sparsity,
        "eval_kl_loss": kl_loss,
        "eval_reg_edge_loss": reg_edge_loss,
        "eval_reg_layer_loss": reg_layer_loss,
    }
    
def freeze_all_except_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def get_optimizers(model, edges_lr, layers_lr, reg_edges_lr, reg_layers_lr, num_training_steps, warmup_steps=0, disable_node_loss=False):
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []

    for n, p in model.named_parameters():
        if 'write_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'read_log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda_edge' in n:
            optimizer_2_group.append(p)
        elif ('sparsity_lambda_node' in n) and (not disable_node_loss):
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
                'lr': layers_lr,
            },
            {
                'params': optimizer_4_group,
                'maximize': True,
                'lr': reg_layers_lr,
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

    raw_datasets = load_datasets(data_args.dataset_path, data_args.max_train_samples, data_args.max_eval_samples, train_split=data_args.train_split)
    n_train = len(raw_datasets["train"])
    
    model = FPT2LMHeadModel.from_pretrained(
        model_args.initialize_from,
        with_embedding_nodes=data_args.with_embedding_nodes,
        disable_linear_regularization_term=data_args.disable_linear_reg_term,
    )
    gpt2_model = FPT2LMHeadModel.from_pretrained(
        "gpt2",
        with_embedding_nodes=data_args.with_embedding_nodes,
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    freeze_all_except_pruning_params(model)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    # Data collator
    collator = DataCollatorCustom(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )
    
    optimizers = get_optimizers(
        model, 
        edges_lr=data_args.edge_learning_rate,
        layers_lr=data_args.layer_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_layers_lr=data_args.reg_layer_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps,
        disable_node_loss=data_args.disable_node_loss
    )

    # Initialize our Trainer
    trainer = FPT2InfoTrainer(
        model=model,
        gpt2_model=gpt2_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_layer_sparsity=data_args.start_layer_sparsity,
        target_layer_sparsity=data_args.target_layer_sparsity,
        skip_layer_loss_if_higher_sparsity=data_args.stop_optimizing_layer_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
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

    kwargs = {"finetuned_from": "gpt-2"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()