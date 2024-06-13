#!/bin/bash -l
#SBATCH --job-name=fs_prune-fllama
#SBATCH --nodes=4 
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr
#SBATCH --gres=gpu:8
#SBATCH --mem=700G
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=16

LOG_DIR=joblog

num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
if [ -z "$SLURM_GPUS_PER_NODE" ]; then
    export SLURM_GPUS_PER_NODE=8
fi
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))
export NUM_NODES=$num_nodes

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

ELR=0.8
LLR=$ELR
RELR=0.4
RLLR=$RELR
EDGE_SPARSITY=1.2
NODE_SPARSITY=0.7
TOTAL=6000
WARMUP=5500
SEED=42
OUTPUT_DIR=./data/runs/fllama-fs-s${SEED}-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/

mkdir -p $OUTPUT_DIR

# Add --with_embedding_nodes if you want to allow the model to prune the embedding nodes
# It should work with the same hyperparameters, but give you a slightly sparser circuit

# Remove --disable_node_loss if you want to prune with node loss

srun bash run_scripts/wrapper_launch_fllama_prune.sh \
src/prune/fllama_boolean_expressions_fs.py \
--report_to wandb \
--do_train \
--dataset_path ./data/datasets/boolean_expressions/ \
--initialize_from meta-llama/CodeLlama-13b-Instruct-hf \
--ref_initialize_from meta-llama/CodeLlama-13b-Instruct-hf \
--max_seq_length 72 \
--per_device_train_batch_size 1 \
--edge_learning_rate $ELR \
--node_learning_rate $LLR \
--reg_edge_learning_rate $RELR \
--reg_node_learning_rate $RLLR \
--max_steps $TOTAL \
--warmup_steps 200 \
--save_steps 512 \
--logging_steps 8 \
--save_total_limit 1 \
--start_edge_sparsity 0.00 \
--target_edge_sparsity $EDGE_SPARSITY \
--start_node_sparsity 0.00 \
--target_node_sparsity $NODE_SPARSITY \
--num_sparsity_warmup_steps $WARMUP \
--max_train_samples 8000 \
--output_dir $OUTPUT_DIR \
--remove_unused_columns false \
--dataloader_num_workers 0 \
--warmup_type linear \
--bf16 \
--gradient_checkpointing \
--seed $SEED \
--disable_node_loss