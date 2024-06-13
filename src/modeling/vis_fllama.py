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
from modeling_fllama import FLlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m1", "--model1", type=str, required=True)
    parser.add_argument("-m2", "--model2", type=str, default="")
    parser.add_argument("-w", "--with_embedding_nodes", action="store_true")
    parser.add_argument("-o", "--output", type=str, default="")
    
    args = parser.parse_args()
    
    if args.output == "":
        if args.model2 == "":
            args.output = os.path.join(args.model1, "edges.json")
        else:
            raise ValueError("Output file must be specified when comparing two models") 
    
    return args

def main():
    args = parse_args()
    
    model = FLlamaForCausalLM.from_pretrained(args.model1, with_embedding_nodes=args.with_embedding_nodes)
    
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

    edge_threshold = threshold
    print("Edge threshold (1):", format(edge_threshold, '.60g'))

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
    node_threshold = threshold
    print("Node threshold (1):", format(node_threshold ,'.60g'))
    
    overall_edge_sparsity = model.get_effective_edge_sparsity()
    print("Overall edge sparsity (1):", format(overall_edge_sparsity))

    edges = model.get_edges()
    
    if args.model2 != "":
        model2 = FLlamaForCausalLM.from_pretrained(args.model2, with_embedding_nodes=args.with_embedding_nodes)
        edge_sparsity = model2.get_edge_sparsity()
        node_sparsity = model2.get_node_sparsity()

        # Binary search for the threshold
        l = 0
        r = 1
        while r-l > 1e-5:
            threshold = (l+r)/2
            model2.set_edge_threshold_for_deterministic(threshold)
            sparsity = model2.get_edge_sparsity()
            if sparsity > edge_sparsity:
                r = threshold
            else:
                l = threshold
        model2.set_edge_threshold_for_deterministic(threshold)

        edge_threshold = threshold
        print("Edge threshold (2):", format(edge_threshold, '.60g'))

        # Binary search for the threshold
        l = 0
        r = 1
        while r-l > 1e-5:
            threshold = (l+r)/2
            model2.set_node_threshold_for_deterministic(threshold)
            sparsity = model2.get_node_sparsity()
            if sparsity > node_sparsity:
                r = threshold
            else:
                l = threshold
        model2.set_node_threshold_for_deterministic(threshold)
        node_threshold = threshold
        print("Node threshold (2):", format(node_threshold ,'.60g'))
        
        overall_edge_sparsity = model.get_effective_edge_sparsity()
        print("Overall edge sparsity (2):", format(overall_edge_sparsity))
        
        edges2 = model2.get_edges()
        
        # Intersection
        edges = [e[0]+"#"+e[1] for e in edges]
        edges2 = [e[0]+"#"+e[1] for e in edges2]
        edges = list(set(edges).intersection(edges2))
        edges = [e.split("#") for e in edges]
    
    print("Saving", len(edges), "edges...")
    json.dump(edges, open(args.output, "w+"), indent=4)

if __name__ == "__main__":
    main()