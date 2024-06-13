EDGE_SPARSITY=1.02
NODE_SPARSITY=0.1
ELR=0.03
LLR=0.03
RELR=0.001
RLLR=0.001
TOTAL=6000
WARMUP=5900

WANDB_MODE=disabled python src/prune/erazr_reverse.py \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/reverse-t3-s3 \
    --initialize_from ./data/tracr_models/reverse.tracr.pkl \
    --seq_length 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --edge_learning_rate $ELR \
    --node_learning_rate $LLR \
    --reg_edge_learning_rate $RELR \
    --reg_node_learning_rate $RLLR \
    --max_steps $TOTAL \
    --warmup_steps 1500 \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --save_steps 64 \
    --logging_steps 4 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_node_sparsity 0.00 \
    --target_node_sparsity $NODE_SPARSITY \
    --num_sparsity_warmup_steps $WARMUP \
    --max_train_samples 100000 \
    --max_eval_samples 100000 \
    --output_dir ./data/runs/erazr-reverse-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --label_names labels \
    --warmup_type linear \
    --zero_ablation \
    --disable_node_loss \
    --overwrite_output_dir