master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo accelerate launch --config_file run_scripts/fsdp_configs/eval_config.yaml \
--main_process_ip ${MASTER_ADDR} \
--main_process_port ${MASTER_PORT} \
--machine_rank ${SLURM_NODEID} \
--num_machines ${NUM_NODES} \
--num_processes ${WORLD_SIZE} $@

accelerate launch --config_file run_scripts/fsdp_configs/eval_config.yaml \
--main_process_ip ${MASTER_ADDR} \
--main_process_port ${MASTER_PORT} \
--machine_rank ${SLURM_NODEID} \
--num_machines ${NUM_NODES} \
--num_processes ${WORLD_SIZE} $@