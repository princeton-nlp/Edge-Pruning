EDGE_SPARSITY=0.92
NODE_SPARSITY=0.4
ELR=1
LLR=1 
RELR=0.0001
RLLR=0.0001
TOTAL=720
WARMUP=640

WANDB_MODE=disabled python src/prune/erazr_xproportion.py \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/xproportion-t4-s4 \
    --initialize_from ./data/tracr_models/xproportion.tracr.pkl \
    --seq_length 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --edge_learning_rate $ELR \
    --node_learning_rate $LLR \
    --reg_edge_learning_rate $RELR \
    --reg_node_learning_rate $RLLR \
    --max_steps $TOTAL \
    --warmup_steps 96 \
    --evaluation_strategy steps \
    --eval_steps 8 \
    --save_steps 8 \
    --logging_steps 4 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_node_sparsity 0.00 \
    --target_node_sparsity $NODE_SPARSITY \
    --num_sparsity_warmup_steps $WARMUP \
    --max_train_samples 100000 \
    --max_eval_samples 100000 \
    --output_dir ./data/runs/erazr-xproportion-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --label_names labels \
    --warmup_type linear \
    --zero_ablation \
    --overwrite_output_dir 