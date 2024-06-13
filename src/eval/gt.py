import os
import json
import argparse
from tqdm import tqdm
from scipy.stats import kendalltau

import torch
from transformers import AutoTokenizer
from datasets import load_from_disk

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)   # Very hacky but the imports are annoying otherwise
from modeling_fpt2 import FPT2LMHeadModel

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

@torch.no_grad()
def load_avg_activations(model, path, device):
    avg_activations = pickle.load(open(path, "rb"))
    model.load_captured_activations(avg_activations.to(device))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", "-m", default="data/runs/example/")
    parser.add_argument("--with-embedding-nodes", "-w", action="store_true") # TRUE if the run allowed removing embedding nodes
                                                                             # Here WITH means that masks were modeled over embedding nodes
    
    parser.add_argument("--sparsity-edge", "-se", default=None, type=float) # If you want to override the sparsity of the model
    parser.add_argument("--sparsity-node", "-sn", default=None, type=float) # If you want to override the sparsity of the model
    parser.add_argument("--data-path", "-d", default="./data/datasets/gt/")
    parser.add_argument("--skip", "-k", default=0, type=int)
    parser.add_argument("--num-examples", "-n", default=1000000, type=int)
    parser.add_argument("--out-path", "-o", default=None)

    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", default=32, type=int)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    info("[i] Loading model and tokenizer...")
    model = FPT2LMHeadModel.from_pretrained(args.model_name_or_path, with_embedding_nodes=args.with_embedding_nodes).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    control_model = FPT2LMHeadModel.from_pretrained("gpt2", with_embedding_nodes=args.with_embedding_nodes).to(args.device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.sparsity_edge is None:
        args.sparsity_edge = model.get_edge_sparsity()
    if args.sparsity_node is None:
        args.sparsity_node = model.get_node_sparsity()

    info("[i] Loading data...")
    data = load_from_disk(args.data_path)["test"]
    if args.num_examples < len(data):
        data = data[:args.num_examples]
    
    info("[i] Preparing data...")
    all_digits = torch.LongTensor([tokenizer.encode("{:02d}".format(i))[0] for i in range(100)]).to(args.device)
    sentences = []
    corr_sentences = []
    digits = []
    for i in tqdm(range(len(data))):
        sentences.append(data[i]['prefix'])
        corr_sentences.append(data[i]['corr_prefix'])
        digits.append(int(data[i]['digits']))

    info(f"[i] {len(sentences)} examples left after filtering.")

    if args.sparsity_edge is not None:
        # Binary search for the threshold
        info("[i] Searching for threshold...")
        l = 0
        r = 1
        while r-l > 1e-5:
            threshold = (l+r)/2
            model.set_edge_threshold_for_deterministic(threshold)
            sparsity = model.get_edge_sparsity()
            if sparsity > args.sparsity_edge:
                r = threshold
            else:
                l = threshold
        info(f"[i] Edge Threshold found: {threshold}")
        info(f"[i] Edge Sparsity: {sparsity}")
    else:
        threshold = args.threshold_edge
        if threshold is None:
            info("[i] No edge threshold specified")
        else:
            info(f"[i] Using edge threshold: {threshold}")
            model.set_edge_threshold_for_deterministic(threshold)
        
    if args.sparsity_node is not None:
        # Binary search for the threshold
        info("[i] Searching for threshold...")
        l = 0
        r = 1
        while r-l > 1e-5:
            threshold = (l+r)/2
            model.set_node_threshold_for_deterministic(threshold)
            sparsity = model.get_node_sparsity()
            if sparsity > args.sparsity_node:
                r = threshold
            else:
                l = threshold
        info(f"[i] Node Threshold found: {threshold}")
        info(f"[i] Node Sparsity: {sparsity}")
    else:
        threshold = args.threshold_node
        if threshold is None:
            info("[i] No node threshold specified")
        else:
            info(f"[i] Using node threshold: {threshold}")
            model.set_node_threshold_for_deterministic(threshold)

    overall_edge_sparsity = model.get_effective_edge_sparsity()
    info(f"[i] Overall Edge Sparsity: {overall_edge_sparsity}")

    info("[i]     Producing outputs...")
    prob_diff = 0
    prob_diff_10 = 0
    bar = tqdm(range(0, len(sentences), args.batch_size))
    count = 0
    all_probs = []
    kt = 0
    kl = 0

    for i in bar:
        indices = []
        n = min(args.batch_size, len(sentences)-i)
        for j in range(n):
            indices.append(tokenizer(sentences[i+j], return_tensors="pt").input_ids.shape[1]-1)
        indices = torch.LongTensor(indices).to(args.device)
        input_ids = tokenizer(sentences[i:i+n], return_tensors="pt", padding=True).input_ids.to(args.device)
        corr_input_ids = tokenizer(corr_sentences[i:i+n], return_tensors="pt", padding=True).input_ids.to(args.device)

        control_outputs = control_model(input_ids)
        corr_x = control_model(corr_input_ids, output_writer_states=True).writer_states
        outputs = model(input_ids, corr_x=corr_x)
        
        logits = torch.gather(
            outputs.logits, 
            1, 
            indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, outputs.logits.shape[-1])
        ).squeeze()
        logits = torch.gather(logits, 1, all_digits.unsqueeze(0).repeat(logits.shape[0], 1))
        probs = torch.nn.functional.softmax(logits, dim=1)
        ref_logits = torch.gather(
            control_outputs.logits,
            1,
            indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, control_outputs.logits.shape[-1])
        ).squeeze()
        ref_logits = torch.gather(ref_logits, 1, all_digits.unsqueeze(0).repeat(ref_logits.shape[0], 1))
        ref_probs = torch.nn.functional.softmax(ref_logits, dim=1)
        
        logits = torch.nn.functional.log_softmax(logits, dim=1)
        ref_logits = torch.nn.functional.log_softmax(ref_logits, dim=1)
        
        for j in range(n):
            prob_diff += (probs[j, digits[i+j]+1:].sum() - probs[j, :digits[i+j]].sum()).detach().cpu().item()
            l = max(0, digits[i+j]-10)
            r = min(100, digits[i+j]+11)
            prob_diff_10 += (probs[j, digits[i+j]+1:r].sum() - probs[j, l:digits[i+j]].sum()).detach().cpu().item()
            probs_ = probs[j].detach().cpu().tolist()
            ref_ = ref_probs[j].detach().cpu().tolist()
            kt_ = kendalltau(probs_, ref_).correlation
            kt += kt_
            all_probs.append(probs_)
        kl += torch.nn.functional.kl_div(logits, ref_logits, reduction="sum", log_target=True).detach().cpu().item()
            
        count += n
        bar.set_description("PD={:.4f} PD10={:.4f} KT={:.4f} KL={:.4f}".format(
            prob_diff/count, prob_diff_10/count, kt/count, kl/count
        ))

    prob_diff /= len(sentences)
    prob_diff_10 /= len(sentences)
    kt /= len(sentences)
    kl /= len(sentences)

    info(f"[i]     Probability difference: {prob_diff}")  
    info(f"[i]     Probability difference (10): {prob_diff_10}")  
    info(f"[i]     Kendall's Tau: {kt}") 
    info(f"[i]     KL Divergence: {kl}")

    if args.out_path is not None:
        info("[i]     Saving outputs...")
        with open(args.out_path, "w+") as f:
            json.dump(all_probs, f, indent=4)

if __name__ == '__main__':
    main()