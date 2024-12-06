"""
Example script to run ACDC with our version of the IOI dataset. This should be placed in a repository cloned from https://github.com/ArthurConmy/Automatic-Circuit-Discovery.
"""

import torch
from tqdm import tqdm
from datasets import load_from_disk
import json
import random
import argparse
import pickle
from transformer_lens.HookedTransformer import HookedTransformer

from acdc.TLACDCExperiment import TLACDCExperiment

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--threshold", "-t", default=0.001, type=float)
    parser.add_argument("--dataset-path", "-d", default="data/datasets/merged/ioi/")
    parser.add_argument("--max-examples", "-n", default=200, type=int)
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--max-num-epochs", "-e", default=10000, type=int)
    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out-path", "-o", default=None)
    parser.add_argument("--out-json-path", "-j", default=None)
    parser.add_argument("--out-pickle-path-initial", "-i", default="outs/ioi-initial.pkl")
    parser.add_argument("--out-pickle-path-final", "-f", default=None)
    
    args = parser.parse_args()
    
    if args.out_path is None:
        args.out_path = f"results/ioi-sweep/ioi-t{args.threshold}-exp.pkl"
    if args.out_json_path is None:
        args.out_json_path = f"results/ioi-sweep/ioi-t{args.threshold}-graph.json"
    if args.out_pickle_path_final is None:
        args.out_pickle_path_final = f"results/ioi-sweep/ioi-t{args.threshold}-graph.pkl"
    
    return args

args = parse_args()

threshold = args.threshold
dataset_path = args.dataset_path
max_examples = args.max_examples
batch_size = args.batch_size
max_num_epochs = args.max_num_epochs
device = args.device
out_path = args.out_path
out_json_path = args.out_json_path
out_pickle_path_initial = args.out_pickle_path_initial
out_pickle_path_final = args.out_pickle_path_final

def get_data(data, tokenizer, max_examples):
    if max_examples < len(data):
        data = data.select(range(max_examples))
    tokens = tokenizer(data["ioi_sentences"], return_tensors="pt", padding=True).input_ids
    corrupted_tokens = tokenizer(data["corr_ioi_sentences"], return_tensors="pt", padding=True).input_ids
    
    cont_indices = []
    correct_tokens = []
    distractor_tokens = []
    for i in tqdm(range(len(data))):
        prefix = data["ioi_sentences"][i]
        prefix = prefix[:prefix.rfind(" ")]
        idx = len(tokenizer.tokenize(prefix)) - 1
        correct_token = tokenizer.encode(" " + data["a"][i])[0]
        distractor_token = tokenizer.encode(" " + data["b"][i])[0]
        correct_tokens.append(correct_token)
        distractor_tokens.append(distractor_token)
        cont_indices.append(idx)
        
    cont_indices = torch.LongTensor(cont_indices)
    correct_tokens = torch.LongTensor(correct_tokens)
    distractor_tokens = torch.LongTensor(distractor_tokens)
    
    return tokens, corrupted_tokens, cont_indices, correct_tokens, distractor_tokens

def logit_difference(
    logits,
    indices,
    correct_tokens,
    distractor_tokens
):
    logits = torch.gather(logits, 1, indices.reshape(-1, 1, 1).repeat(1, 1, logits.shape[-1])).squeeze()
    correct_logits = torch.gather(logits, 1, correct_tokens.reshape(-1, 1)).squeeze()
    distractor_logits = torch.gather(logits, 1, distractor_tokens.reshape(-1, 1)).squeeze()
    
    return (correct_logits - distractor_logits).mean()

def get_logprobs(
    logits,
    indices,
):
    logits = torch.gather(logits, 1, indices.reshape(-1, 1, 1).repeat(1, 1, logits.shape[-1])).squeeze()
    logits = torch.log_softmax(logits, dim=-1)
    
    return logits

@torch.no_grad()
def pred(model, tokens, device, batch_size=32):
    logits = []
    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        logits_ = model(tokens[i:i+batch_size].to(device)).cpu()
        logits.append(logits_)
    logits = torch.cat(logits, dim=0)
    return logits

# Seed so that we can compare results
random.seed(42)
torch.random.manual_seed(42)

torch.autograd.set_grad_enabled(False)

model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk(dataset_path)
train_toks, train_corr_toks, train_idx, train_correct, train_distractor = get_data(dataset["train"], tokenizer, max_examples)

with torch.no_grad():
    train_pred = pred(model, train_toks, device)
    train_corr_pred = pred(model, train_corr_toks, device, batch_size)

def kl_divergence(
    logits,
    full_model_logits,
    indices,
    start=None,
    end=None,
):
    if start is None:
        start = 0
    if end is None:
        end = logits.shape[0]
    logits = logits[start:end]
    indices = indices[start:end].to(logits.device)
    full_model_logits = full_model_logits[start:end].to(logits.device)
    
    logits = torch.gather(
        logits, 
        1, 
        indices.reshape(-1, 1, 1).repeat(1, 1, logits.shape[-1])
    ).squeeze()
    if full_model_logits.dim() == 3:
        full_model_logits = torch.gather(
            full_model_logits, 
            1, 
            indices.reshape(-1, 1, 1).repeat(1, 1, full_model_logits.shape[-1])
        ).squeeze()
        
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    full_model_logits = torch.nn.functional.log_softmax(full_model_logits, dim=-1)
    kl_div = torch.nn.functional.kl_div(logits, full_model_logits, log_target=True, reduction="batchmean")
    
    return kl_div

def get_targets(tokens, indices):
    indices_ = indices + 1
    return torch.gather(tokens, 1, indices_.reshape(-1, 1)).squeeze()

def get_preds(logits, indices):
    logits = torch.gather(logits, 1, indices.reshape(-1, 1, 1).repeat(1, 1, logits.shape[2])).squeeze()
    preds = torch.argmax(logits, dim=-1)
    return preds

def accuracy(preds, targets):
    return (preds == targets).float().mean().item()

@torch.no_grad()
def pred(model, tokens, device, batch_size=32):
    logits = []
    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        logits_ = model(tokens[i:i+batch_size].to(device)).cpu()
        logits.append(logits_)
    logits = torch.cat(logits, dim=0)
    return logits

metric = lambda logits: kl_divergence(logits, train_pred, train_idx)

model.reset_hooks()
experiment = TLACDCExperiment(
    model=model, 
    ds=train_toks,
    ref_ds=train_corr_toks,
    threshold=threshold,
    metric=metric,
    online_cache_cpu=False, # Use GPU
    corrupted_cache_cpu=False, # Use GPU
    verbose=True
)

bar = tqdm(range(max_num_epochs))
for i in bar:
    experiment.step()
    edge_count = experiment.count_no_edges()
    bar.set_description(f"Epoch {i+1}: {edge_count} edges")
    
    if i == 0:
        experiment.save_edges(out_pickle_path_initial)
    if experiment.current_node is None:
        break
    
experiment.save_edges(out_pickle_path_final)

graph = experiment.save_subgraph(return_it=True)
pickle.dump(graph, open(out_path, "wb"))

# This graph is super unweildy, make it better
good_graph = []
good_graph_extra = []
for to_name, to_idx, from_name, from_idx in graph:
    to_parts = to_name.split(".")
    from_parts = [from_name] if "." not in from_name else from_name.split(".")
    if from_parts[0] in ["hook_embed", "hook_pos_embed"]:
        print(f"Skipping {from_name}->{to_name} and adding to extra as always exists")
        continue
    
    to_layer_num = int(to_parts[1])
    from_layer_num = int(from_parts[1])
    if to_parts[2] == "attn":
        # These should be skipped, the attn input ones don't have 'attn'
        # To be safe, add some assertions
        if to_parts[3] == "hook_result":
            assert from_parts[3] in ["hook_q", "hook_k", "hook_v"], f"{from_name}->{to_name}! (1)"
            assert to_layer_num == from_layer_num and from_idx[2] == to_idx[2], f"{from_name}->{to_name}! (2)"
            print(f"Skipping {from_name}->{to_name}")
        elif to_parts[3] in ["hook_q", "hook_k", "hook_v"]:
            assert from_parts[2] in ["hook_q_input", "hook_k_input", "hook_v_input"], f"{from_name}->{to_name}! (3)"
            assert to_layer_num == from_layer_num and from_idx[2] == to_idx[2] and from_parts[2].startswith(to_parts[3]), \
                f"{from_name}->{to_name}! (4)"
            print(f"Skipping {from_name}->{to_name} and adding to extra as always exists")
        else:
            raise ValueError(f"Unknown to: {to_name}")
        good_graph_extra.append({
            "from": from_name,
            "to": to_name,
            "head_num": to_idx[2],
        })
        continue
    elif to_parts[2] == "hook_mlp_out":
        assert from_parts[2] == "hook_mlp_in" and from_layer_num == to_layer_num, f"{from_name}->{to_name}! (5)"
        print(f"Skipping {from_name}->{to_name} and adding to extra as always exists")
        good_graph_extra.append({
            "from": from_name,
            "to": to_name,
        })
        continue
    elif to_parts[2] == "hook_resid_post":
        assert to_layer_num == 11, f"{from_name}->{to_name}! (6)"
        to_name = "resid_post"
    elif to_parts[2] == "hook_mlp_in":
        to_name = f"mlp.{to_layer_num}"
    elif to_parts[2] == "hook_q_input":
        head_num = to_idx[2]
        to_name = f"head.{to_layer_num}.{head_num}.q"
    elif to_parts[2] == "hook_k_input":
        head_num = to_idx[2]
        to_name = f"head.{to_layer_num}.{head_num}.k"
    elif to_parts[2] == "hook_v_input":
        head_num = to_idx[2]
        to_name = f"head.{to_layer_num}.{head_num}.v"
    else:
        print(from_name, to_name)
        raise ValueError(f"Unknown to: {to_name}")
    
    # From should not have resid_posts -- only hook_result and hook_mlp_out
    if from_parts[2] == "attn":
        assert from_parts[3] == "hook_result", f"{from_name}->{to_name}! (7)"
        head_num = from_idx[2]
        from_name = f"head.{from_layer_num}.{head_num}"
    elif from_parts[2] == "hook_mlp_out":
        from_name = f"mlp.{from_layer_num}"
    elif from_parts[2] == "hook_resid_pre":
        assert from_layer_num == 0, f"{from_name}->{to_name}! (8)"
        print(f"Skipping {from_name}->{to_name} and adding to extra as always exists")
        good_graph_extra.append({
            "from": from_name,
            "to": to_name,
        })
        continue
    else:
        print(from_name, to_name)
        raise ValueError(f"Unknown from: {from_name}")
    
    good_graph.append({
        "from": from_name,
        "to": to_name,
    })

print("Final")
print(good_graph)
print(f"No. edges: {len(good_graph)}, extra: {len(good_graph_extra)}")

combined_graph = {
    "original": good_graph,
    "extra": good_graph_extra,
}

json.dump(combined_graph, open(out_json_path, "w+"), indent=4)