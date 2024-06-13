import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk

from accelerate import Accelerator

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)
from modeling_fllama import FLlamaForCausalLM

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def info(text):
    print(f"{bcolors.OKBLUE}{text}{bcolors.ENDC}")

def good(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")

def bad(text):
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", "-m", default="meta-llama/CodeLlama-13b-Instruct-hf")
    parser.add_argument("--edges", "-e", default=None)
    parser.add_argument("--reference_model", "-r", default="meta-llama/CodeLlama-13b-Instruct-hf")
    parser.add_argument("--dataset-path", "-d", default="data/datasets/boolean_expressions")
    parser.add_argument("--batch-size", "-b", default=1, type=int)
    parser.add_argument("--skip", "-k", default=0, type=int)
    parser.add_argument("--num-examples", "-n", default=400, type=int)
    parser.add_argument("--mode", "-M", default="instruction", choices=["instruction", "fewshot", "zeroshot"])
    parser.add_argument("--bf16", "-bf16", default=False, action="store_true")
    parser.add_argument("--tag", "-t", default=None)
    parser.add_argument("--with-embedding-nodes", "-w", default=False, action="store_true")
    
    args = parser.parse_args()
    
    return args

@torch.no_grad()
def main():
    args = parse_args()
    
    accelerator = Accelerator()
    
    info("[i] Loading models and tokenizer...")
    if args.bf16:
        model = FLlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, with_embedding_nodes=args.with_embedding_nodes)
    else:
        model = FLlamaForCausalLM.from_pretrained(args.model_name_or_path, with_embedding_nodes=args.with_embedding_nodes)
    
    if args.model_name_or_path.startswith("meta-llama/"):
        model.reset_all_log_alphas()    # Not strictly necessary, but reset anyway
    else:
        if args.edges is not None:
            edges = json.load(open(args.edges))
            model.load_all_log_alphas(edges)
            
            # The edges are deterministic, so we can set the thresholds to 0
            model.set_edge_threshold_for_deterministic(0)
            model.set_node_threshold_for_deterministic(0)
        else:
            edge_sparsity = model.get_edge_sparsity()
            node_sparsity = model.get_node_sparsity()

            # Binary search for the threshold
            l = 0
            r = 1
            while r-l > 1e-5:
                threshold = (l+r)/2
                model.set_edge_threshold_for_deterministic(threshold)
                sparsity = model.get_edge_sparsity()
                if sparsity > edge_sparsity:
                    r = threshold
                else:
                    l = threshold
            model.set_edge_threshold_for_deterministic(threshold)

            # Binary search for the threshold
            l = 0
            r = 1
            while r-l > 1e-5:
                threshold = (l+r)/2
                model.set_node_threshold_for_deterministic(threshold)
                sparsity = model.get_node_sparsity()
                if sparsity > node_sparsity:
                    r = threshold
                else:
                    l = threshold
            model.set_node_threshold_for_deterministic(threshold)

            overall_edge_sparsity = model.get_effective_edge_sparsity()
            info(f"[i] Overall edge sparsity: {overall_edge_sparsity:.5f}")
            edges = model.get_edges()
            info(f"[i] {len(edges)} edges overall")            
    
    model = accelerator.prepare(model)
    
    if args.bf16:
        ref_model = FLlamaForCausalLM.from_pretrained(args.reference_model, torch_dtype=torch.bfloat16, with_embedding_nodes=args.with_embedding_nodes)
    else:
        ref_model = FLlamaForCausalLM.from_pretrained(args.reference_model, with_embedding_nodes=args.with_embedding_nodes)
    ref_model = accelerator.prepare(ref_model)
    ref_model.reset_all_log_alphas()
    
    tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    info("[i] Loading dataset...")
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)["test"]
    else:
        dataset = load_dataset(args.dataset_path, args.subset)["test"]
    end_idx = min(args.skip + args.num_examples, len(dataset))
    dataset = dataset.select(range(args.skip, end_idx))
    
    def format_fewshot(entry):
        shot1 = "(True and False) or (False and True) is False"
        shot2 = "(True or (not True)) and False is False"
        shot3 = "(not (True and False)) or (False and True) is True"
        ip = f"{shot1}\n{shot2}\n{shot3}\n{entry['input']}"
        corr_ip = f"{shot1}\n{shot2}\n{shot3}\n{entry['corr_input']}"
        return ip, corr_ip, entry["target"].strip()
    
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
    
    def format_zeroshot(entry):
        ip = entry["input"]
        corr_ip = entry["corr_input"]
        return ip, corr_ip, entry["target"].strip()
    
    format_map = {
        "fewshot": format_fewshot,
        "zeroshot": format_zeroshot,
        "instruction": format_instruction,
    }
    
    format_fn = format_map[args.mode]
    
    info("[i] Preparing data...")
    prefixes = []
    corr_prefixes = []
    indices = []
    targets = []
    for entry in dataset:
        ip, corr_ip, target = format_fn(entry)
        prefixes.append(ip)
        corr_prefixes.append(corr_ip)
        indices.append(len(tokenizer.encode(ip))-1)
        targets.append(target)
    
    max_l = 0
    
    vocab_size = model.config.vocab_size
    
    n = 0
    accuracy = 0
    kl_divergence = 0
    exact_match = 0
    
    bar = tqdm(range(0, len(prefixes), args.batch_size))
    for i in bar:
        i_to = min(i+args.batch_size, len(prefixes))
        corr_inputs = tokenizer(
            corr_prefixes[i:i_to],
            return_tensors="pt",
            padding=True,  
        ).input_ids.to(accelerator.device)
        
        corr_x = ref_model(corr_inputs, output_writer_states=True).writer_states
        
        inputs = tokenizer(
            prefixes[i:i_to], 
            return_tensors="pt", 
            padding=True, 
        ).input_ids.to(accelerator.device)
        max_l = max(max_l, inputs.shape[1])
        
        idx_select = torch.LongTensor(indices[i:i_to]).to(accelerator.device)
        idx_select = idx_select.reshape(-1, 1, 1).repeat(1, 1, vocab_size)
        
        ref_logits = ref_model(inputs).logits
        ref_logits = torch.gather(
            ref_logits,
            1,
            idx_select
        )
        ref_logits = ref_logits.reshape(ref_logits.shape[0], -1)
        
        outputs = model(inputs, corr_x=corr_x)
        logits = outputs.logits
        logits = torch.gather(
            logits,
            1,
            idx_select
        )
        logits = logits.reshape(logits.shape[0], -1)   
        
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        ref_logits = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        
        kl = torch.nn.functional.kl_div(
            logits,
            ref_logits,
            reduction="batchmean",
            log_target=True
        ).cpu().item()
        
        preds = logits.argmax(-1).cpu().tolist()
        preds = [tokenizer.decode(p).strip() for p in preds]
        ref_preds = ref_logits.argmax(-1).cpu().tolist()
        ref_preds = [tokenizer.decode(p).strip() for p in ref_preds]
        
        n_ = i_to - i
        n += n_
        kl_divergence += kl
        for j in range(n_):
            if preds[j] == targets[i+j]:
                accuracy += 1
            if preds[j] == ref_preds[j]:
                exact_match += 1
    
        es = outputs.model_edge_sparsity.cpu().item()
        ns = outputs.model_node_sparsity.cpu().item()
        bar.set_description(f"Accuracy: {accuracy/n:.2f}, KL: {kl_divergence/n:.2f}, EM: {exact_match/n:.2f} ES: {es:.2f}, NS: {ns:.2f}")
    
    accuracy /= n
    kl_divergence /= n
    exact_match /= n
    
    info(f"[i] Accuracy: {accuracy}")
    info(f"[i] KL divergence: {kl_divergence}")
    info(f"[i] Exact match: {exact_match}")

if __name__ == '__main__':
    main()