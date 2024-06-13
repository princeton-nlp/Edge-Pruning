#!/bin/bash
#SBATCH --job-name=eval-fllama
#SBATCH --nodes=1 
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --time=1:30:00

num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
if [ -z "$SLURM_GPUS_PER_NODE" ]; then
    export SLURM_GPUS_PER_NODE=4
fi
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))
export NUM_NODES=$num_nodes

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

MODEL="/path/to/pruned/model"
REFERENCE="meta-llama/CodeLlama-13b-Instruct-hf"
BATCH_SIZE=4
MODE="instruction" # "fewshot" 

# If you want to evaluate the intersection, or a custom set of edges
## Step 1: Obtain the intersection
# python src/modeling/vis_fllama.py -m1 /path/to/model1 -m2 /path/to/model2 -o /path/to/output.json
## Step 2: Run evaluation with the additional flag -e /path/to/output.json
# For this step $MODEL should point to $REFERENCE or either of model1/model2

srun bash run_scripts/wrapper_launch_fllama_eval.sh \
src/eval/boolean_expressions.py -m $MODEL -r $REFERENCE -b $BATCH_SIZE -M $MODE -bf16 