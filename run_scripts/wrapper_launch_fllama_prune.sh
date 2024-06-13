master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo WANDB_MODE=disabled accelerate launch --config_file run_scripts/fsdp_configs/prune_config.yaml \
--main_process_ip ${MASTER_ADDR} \
--main_process_port ${MASTER_PORT} \
--machine_rank ${SLURM_NODEID} \
--num_machines ${NUM_NODES} \
--num_processes ${WORLD_SIZE} $@

WANDB_MODE=disabled accelerate launch --config_file run_scripts/fsdp_configs/prune_config.yaml \
--main_process_ip ${MASTER_ADDR} \
--main_process_port ${MASTER_PORT} \
--machine_rank ${SLURM_NODEID} \
--num_machines ${NUM_NODES} \
--num_processes ${WORLD_SIZE} $@