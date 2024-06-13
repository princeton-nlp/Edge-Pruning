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
    parser.add_argument("--data-path", "-d", default="./data/datasets/gp/")
    parser.add_argument("--num-examples", "-n", default=100000, type=int)

    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", default=16, type=int)
    parser.add_argument("--out-path", "-o", default=None)

    args = parser.parse_args()
    
    if args.out_path == "None":
        args.out_path = None

    return args

@torch.no_grad()
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
    sentences = []
    corr_sentences = []
    targets = []
    distractors = []
    for i in tqdm(range(len(data))):
        sentence = data[i]['prefix'] + " " + data[i]['pronoun']
        corr_sentence = data[i]['corr_prefix'] + " " + data[i]['corr_pronoun']
        target = data[i]['pronoun'].lower().strip()
        distractor = "he" if target == "she" else "she"

        target_tokenized = tokenizer.encode(" "+target)
        distractor_tokenized = tokenizer.encode(" "+distractor)

        if len(target_tokenized) > 1 or len(distractor_tokenized) > 1:
            continue    

        sentences.append(sentence)
        corr_sentences.append(corr_sentence)
        targets.append(target_tokenized[0])
        distractors.append(distractor_tokenized[0])

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
        
    accuracy = 0
    logit_difference = 0
    kl_divergence = 0
    exact_match = 0
    outputs_ = []
    info("[i] Producing outputs...")
    bar = tqdm(range(0, len(sentences), args.batch_size))
    for i in bar:
        input_ids = tokenizer(sentences[i:i+args.batch_size], return_tensors="pt", padding=True).input_ids.to(args.device)
        corr_input_ids = tokenizer(corr_sentences[i:i+args.batch_size], return_tensors="pt", padding=True).input_ids.to(args.device)
        prefix_lengths = []
        for j in range(input_ids.shape[0]):
            sentence = sentences[i+j]
            sentence_prefix = sentence[:sentence.rfind(" ")]
            prefix_length = tokenizer(sentence_prefix, return_tensors="pt").input_ids.shape[1]
            prefix_lengths.append(prefix_length)
        
        control_outputs = control_model(input_ids)
        corr_x = control_model(corr_input_ids, output_writer_states=True).writer_states
        outputs = model(input_ids, corr_x=corr_x)
        logits = outputs.logits
        control_logits = control_outputs.logits

        for j in range(input_ids.shape[0]):
            prefix_length = prefix_lengths[j]
            logit_target = logits[j, prefix_length-1, targets[i+j]].detach().cpu().item()
            logit_distractor = logits[j, prefix_length-1, distractors[i+j]].detach().cpu().item()
            logit_difference += logit_target - logit_distractor
            chosen_word = tokenizer.decode(torch.argmax(logits[j, prefix_length-1]).item())
            
            logits_ = torch.nn.functional.log_softmax(logits[j, prefix_length-1], dim=-1)
            control_logits_ = torch.nn.functional.log_softmax(control_logits[j, prefix_length-1], dim=-1)
            kld = torch.nn.functional.kl_div(logits_, control_logits_, reduction="sum", log_target=True)
            kl_divergence += kld.detach().cpu().item()
                        
            choice = torch.argmax(logits[j, prefix_length-1])
            accuracy += (choice == targets[i+j]).int().detach().cpu().item()
            control_choice = torch.argmax(control_logits[j, prefix_length-1])
            exact_match += (choice == control_choice).int().detach().cpu().item()
            
            outputs_.append({
                "sentence": sentences[i+j],
                "target": tokenizer.decode(targets[i+j]),
                "distractor": tokenizer.decode(distractors[i+j]),
                "chosen_word": chosen_word,
                "logit_target": logit_target,
                "logit_distractor": logit_distractor,
                "logit_difference": logit_target - logit_distractor,
                "choice": choice.item(),
            })
            
        bar.set_description(f"Acc: {accuracy/(i+input_ids.shape[0]):.3f}, LD: {logit_difference/(i+input_ids.shape[0]):.3f}")

    accuracy /= len(sentences)
    logit_difference /= len(sentences)
    kl_divergence /= len(sentences)
    exact_match /= len(sentences)

    info(f"[i]     Accuracy: {accuracy}")
    info(f"[i]     Logit difference: {logit_difference}")   
    info(f"[i]     KL Divergence: {kl_divergence}")
    info(f"[i]     Exact Match: {exact_match}")
    
    if args.out_path is not None:
        info(f"[i] Saving outputs to {args.out_path}...")
        json.dump(outputs_, open(args.out_path, "w+"), indent=4)     

if __name__ == '__main__':
    main()