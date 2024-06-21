import os
import json
import argparse

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)
from modeling_fpt2 import FPT2LMHeadModel

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_path", "-i", type=str, required=True)
    parser.add_argument("--out_path", "-o", type=str, default=None)
    parser.add_argument("--with_embedding_nodes", "-w", action="store_true")
    parser.add_argument("--edge_sparsity", "-e", type=float, default=None)
    parser.add_argument("--node_sparsity", "-n", type=float, default=None)

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.join(args.in_path, "edges.json")
        print(f"Output path not specified. Saving to {args.out_path}.")

    return args

def main():
    args = parse_args()

    model = FPT2LMHeadModel.from_pretrained(args.in_path, with_embedding_nodes=args.with_embedding_nodes)
    if args.edge_sparsity is None:
        args.edge_sparsity = model.get_edge_sparsity()
    if args.node_sparsity is None:
        args.node_sparsity = model.get_node_sparsity()
                
    l = 0
    r = 1
    while r-l > 1e-5:
        threshold = (l+r)/2
        model.set_edge_threshold_for_deterministic(threshold)
        sparsity = model.get_edge_sparsity()
        if sparsity > args.edge_sparsity:
            r = threshold
        else:
            l = threshold        

    l = 0
    r = 1
    while r-l > 1e-5:
        threshold = (l+r)/2
        model.set_node_threshold_for_deterministic(threshold)
        sparsity = model.get_node_sparsity()
        if sparsity > args.node_sparsity:
            r = threshold
        else:
            l = threshold

    overall_edge_sparsity = model.get_effective_edge_sparsity()
    print("Overall edge sparsity:", overall_edge_sparsity.item())

    edges = model.get_edges()
                
    json.dump(edges, open(args.out_path, "w+"), indent=4)

if __name__ == '__main__':
    main()